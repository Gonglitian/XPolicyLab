"""Prepare aligned absolute EEF labels, without changing checkpoints or losses.

LIBERO: restore each recorded simulator state, query its native OSC target and
render the matching pre-action images. RoboCasa: reconstruct OSC targets from
the documented LeRobot state/action fields, including mobile-controller history.
The output is an explicit aligned data schema; do not apply the legacy handler's
first-image drop or trajectory shift to it.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def rot6d(matrix):
    """Column-stacked layout used by this checkout's X-VLA LIBERO deployment."""
    return np.concatenate((matrix[..., :, 0],matrix[..., :, 1]),axis=-1)


def from_rot6d(value):
    first=value[..., :3];second=value[..., 3:]
    first=first/np.linalg.norm(first,axis=-1,keepdims=True)
    second=second-first*np.sum(first*second,axis=-1,keepdims=True)
    second=second/np.linalg.norm(second,axis=-1,keepdims=True)
    return np.stack((first,second,np.cross(first,second)),axis=-1)


def write_episode(path,positions,rotations,grippers,proprio,raw_actions,instruction,**extra):
    absolute=np.concatenate((positions,rot6d(rotations),(grippers>0).astype(float)[:,None]),axis=1)
    assert np.isfinite(absolute).all()
    rot_error=float(np.max(abs(from_rot6d(absolute[:,3:9])-rotations)))
    assert rot_error<1e-6
    with h5py.File(path,'a') as f:
        f.create_dataset('abs_action_6d',data=absolute)
        f.create_dataset('action_eef_pose_xyzw',data=np.concatenate((positions,Rotation.from_matrix(rotations).as_quat()),axis=1))
        f.create_dataset('action_eef_rotation_matrix',data=rotations)
        f.create_dataset('proprio_6d',data=proprio)
        f.create_dataset('raw_actions',data=raw_actions)
        f.create_dataset('frame_index',data=np.arange(len(absolute)))
        f.create_dataset('instruction',data=instruction)
        for name,value in extra.items():f.create_dataset(name,data=value)
        f.attrs.update(schema='xvla-aligned-absolute-v1',rotation_6d_layout='R[:,0] followed by R[:,1]',
            gripper='0=open, 1=closed',time_alignment='observation[t] precedes absolute target[t]',
            source_episode_frames=len(absolute),fps=20)
    return rot_error


def prepare_libero(args):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml
    from XPolicyLab.utils.process_data import encode_image_bit,decode_image_bit
    task=benchmark.get_benchmark_dict()['libero_spatial']().get_task(0)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                          camera_heights=128,camera_widths=128)
    env.reset()
    entries=[]
    assets_root=Path(get_libero_path('assets'))
    with h5py.File(args.source) as source:
        instruction=json.loads(source['data'].attrs['problem_info'])['language_instruction']
        controller_config=json.loads(source['data'].attrs['env_args'])['env_kwargs']['controller_configs']
        for eid in args.episodes:
            demo=source[f'data/demo_{eid}'];actions=demo['actions'][:];count=len(actions)
            xml=postprocess_model_xml(demo.attrs['model_file'],{})
            # Original released demos still use LIBERO's old "chiliocosm" name.
            tree=ET.fromstring(xml)
            for element in tree.iter():
                value=element.get('file')
                if value and '/chiliocosm/assets/' in value:
                    relocated=assets_root/value.split('/chiliocosm/assets/',1)[1]
                    if not relocated.is_file():raise FileNotFoundError(relocated)
                    element.set('file',str(relocated))
            xml=ET.tostring(tree,encoding='unicode')
            env.reset_from_xml_string(xml);env.env.sim.reset()
            path=args.output/f'episode_{eid:06d}.hdf5'
            if path.exists():raise FileExistsError(path)
            pos=[];rot=[];prop=[];max_goal_error=0.;max_torque_error=0.
            observation_offset=[]
            with h5py.File(path,'w') as dest:
                dt=h5py.vlen_dtype(np.dtype('uint8'))
                cameras={name:dest.create_dataset('images/'+name,(count,),dtype=dt) for name in ['agentview','wrist']}
                for t,action in enumerate(actions):
                    # Original LIBERO images/obs are post-action (create_dataset.py).
                    # Regenerate pre-action observations, not a guessed one-frame shift.
                    obs=env.set_init_state(demo['states'][t])
                    c=env.env.robots[0].controller;c.update(force=True)
                    if t==0:c.reset_goal()
                    measured=np.concatenate((c.ee_pos,rot6d(c.ee_ori_mat),[0.]))
                    prop.append(measured)
                    observation_offset.append(float(np.linalg.norm(c.ee_pos-demo['obs/ee_pos'][t])))
                    # Match the already tested X-VLA inference image transforms.
                    images={'agentview':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),
                            'wrist':np.ascontiguousarray(obs['robot0_eye_in_hand_image'])}
                    for name,im in images.items():
                        encoded=encode_image_bit(im,quality=95)
                        # Storage view only; image decoding still uses the shared helper.
                        cameras[name][t]=np.frombuffer(bytes(encoded),dtype=np.uint8)
                        if t==0:assert decode_image_bit(encoded).shape==im.shape
                    c.use_delta=True;c.set_goal(action[:6])
                    goal_pos=c.goal_pos.copy();goal_rot=c.goal_ori.copy()
                    delta_torque=c.run_controller().copy()
                    # Same state + absolute target must produce the same arm torques.
                    c.use_delta=False;c.set_goal(np.r_[goal_pos,Rotation.from_matrix(goal_rot).as_rotvec()])
                    max_goal_error=max(max_goal_error,float(np.max(abs(c.goal_ori-goal_rot))),float(np.max(abs(c.goal_pos-goal_pos))))
                    max_torque_error=max(max_torque_error,float(np.max(abs(c.run_controller()-delta_torque))))
                    c.use_delta=True
                    # Preserve original target orientation when zero rotation keeps it.
                    c.goal_pos=goal_pos.copy();c.goal_ori=goal_rot.copy()
                    pos.append(goal_pos);rot.append(goal_rot)
                dest.attrs.update(frame='world',source=str(args.source),source_demo=f'demo_{eid}',
                    images='pre-action RGB rendered from recorded state; X-VLA LIBERO view transforms already applied')
            assert max_goal_error<1e-7 and max_torque_error<1e-5,(eid,max_goal_error,max_torque_error)
            rot_error=write_episode(path,np.array(pos),np.array(rot),actions[:,-1],np.array(prop),actions,instruction)
            entry={'episode':eid,'frames':count,'file':str(path),'sha256':digest(path),
                   'max_goal_roundtrip_error':max_goal_error,'max_delta_absolute_torque_difference':max_torque_error,
                   'rot6d_roundtrip_error':rot_error,'max_original_obs_vs_pre_action_position_difference':max(observation_offset)}
            entries.append(entry);print(json.dumps(entry),flush=True)
        controller_path=Path(inspect.getfile(c.__class__))
    env.close()
    return {'benchmark':'LIBERO','source':str(args.source),'source_sha256':digest(args.source),'episodes':entries,
            'frame':'world','controller_config':controller_config,'controller_source':str(controller_path),
            'controller_sha256':digest(controller_path),'validation':'All frames: native delta vs absolute target and arm torque equivalence',
            'alignment':'Render observation from source states[t], target native controller goal for actions[t]; no index shift',
            'important':'Do not use legacy LiberoHandler first-frame drop on this explicitly aligned export.'}


def prepare_robocasa(args):
    import pandas as pd
    import gymnasium as gym
    import robocasa
    env=gym.make('robocasa/OpenDrawer',split='pretrain',seed=0)
    obs,_=env.reset(seed=0);robot=env.unwrapped.env.robots[0];c=robot.part_controllers['right']
    c.update(force=True)
    # LeRobot stores the historical BODY quaternion, OSC controls the TOOL SITE.
    body_world=robot.sim.data.get_body_xmat(robot.robot_model.eef_name['right'])
    tool_world=robot.sim.data.site_xmat[robot.eef_site_id['right']].reshape(3,3)
    body_to_tool=body_world.T@tool_world
    observed_base_rotation=Rotation.from_quat(obs['state.end_effector_rotation_relative']).as_matrix()
    calibration_error=float(np.max(abs(observed_base_rotation@body_to_tool-c.goal_origin_to_eef_pose()[:3,:3])))
    position_error=float(np.max(abs(obs['state.end_effector_position_relative']-c.world_to_origin_frame(c.ref_pos))))
    assert calibration_error<1e-6 and position_error<1e-6
    assert c.input_ref_frame=='base' and c.impedance_mode=='fixed'
    scales=c.output_max.copy();limits=[c.input_min.copy(),c.input_max.copy()]
    controller_path=Path(inspect.getfile(c.__class__))
    # Feed recorded observable values to the real OSC methods for an independent
    # numerical oracle. No dynamics replay is claimed for these LeRobot files.
    c.update=lambda *a,**kw:None
    tasks={x['task_index']:x['task'] for x in map(json.loads,(args.source/'meta/tasks.jsonl').read_text().splitlines())}
    entries=[]
    for eid in args.episodes:
        source=args.source/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
        df=pd.read_parquet(source);states=np.stack(df['observation.state']);actions=np.stack(df['action'])
        if actions[0,4]>0:raise ValueError(f'Episode {eid} begins in desired-goal mode; initial hidden goal unavailable')
        pos=[];rot=[];prop=[];world_pose=[];maximum_error=0.;oracle_error=0.;max_clip=0.
        c.goal_pos=None;c.goal_ori=None
        for t,(state,action) in enumerate(zip(states,actions)):
            base_rotation=Rotation.from_quat(state[3:7]).as_matrix()
            current_pos=state[7:10];current_rot=Rotation.from_quat(state[10:14]).as_matrix()@body_to_tool
            delta=np.clip(action[5:11],limits[0],limits[1])*scales
            max_clip=max(max_clip,float(np.max(abs(action[5:11]-np.clip(action[5:11],limits[0],limits[1])))))
            desired=action[4]>0
            anchor_pos=pos[-1] if desired else current_pos
            anchor_rot=rot[-1] if desired else current_rot
            goal_pos=anchor_pos+delta[:3]
            goal_rot=Rotation.from_rotvec(delta[3:]).as_matrix()@anchor_rot
            prop.append(np.r_[current_pos,rot6d(current_rot),0.])
            # Actual controller computation, using the recorded base/body state.
            c.origin_pos=state[:3];c.origin_ori=base_rotation
            c.ref_pos=state[:3]+base_rotation@current_pos;c.ref_ori_mat=base_rotation@current_rot
            c.input_type='delta';c.set_goal_update_mode('desired' if desired else 'achieved')
            c.set_goal(action[5:11])
            oracle_error=max(oracle_error,float(np.max(abs(c.goal_pos-goal_pos))),float(np.max(abs(c.goal_ori-goal_rot))))
            # Validate inverse against effective (clipped/scaled) commands.
            restored=np.r_[(goal_pos-anchor_pos)/scales[:3],Rotation.from_matrix(goal_rot@anchor_rot.T).as_rotvec()/scales[3:]]
            maximum_error=max(maximum_error,float(np.max(abs(restored-np.clip(action[5:11],limits[0],limits[1])))))
            c.input_type='absolute';c.set_goal(np.r_[goal_pos,Rotation.from_matrix(goal_rot).as_rotvec()])
            oracle_error=max(oracle_error,float(np.max(abs(c.goal_pos-goal_pos))),float(np.max(abs(c.goal_ori-goal_rot))))
            c.goal_pos=goal_pos.copy();c.goal_ori=goal_rot.copy()
            pos.append(goal_pos);rot.append(goal_rot)
            world_pose.append(np.r_[state[:3]+base_rotation@goal_pos,Rotation.from_matrix(base_rotation@goal_rot).as_quat()])
        assert oracle_error<1e-6 and maximum_error<1e-6,(oracle_error,maximum_error)
        path=args.output/f'episode_{eid:06d}.hdf5'
        if path.exists():raise FileExistsError(path)
        instruction=tasks[int(df['annotation.human.task_description'].iloc[0])]
        rot_error=write_episode(path,np.array(pos),np.array(rot),actions[:,11],np.array(prop),actions,instruction,
            base_motion=actions[:,:4],control_mode=actions[:,4:5],source_state=states,
            action_eef_world_pose_xyzw=np.array(world_pose))
        video_paths={key:str(args.source/f'videos/chunk-{eid//1000:03d}/observation.images.{key}/episode_{eid:06d}.mp4')
                     for key in ['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']}
        assert all(Path(p).is_file() for p in video_paths.values())
        with h5py.File(path,'a') as f:
            f.attrs.update(frame='robot base center (absolute target in this frame, not a delta)',source=str(source),
                           video_paths=json.dumps(video_paths),video_alignment='video frame t -> action target t',
                           auxiliary_controls='base_motion(4) and signed control_mode(1), preserved unchanged; not EEF channels')
            assert np.array_equal(f['base_motion'][:],actions[:,:4]) and np.array_equal(f['control_mode'][:],actions[:,4:5])
        entry={'episode':eid,'frames':len(actions),'file':str(path),'sha256':digest(path),'source_sha256':digest(source),
               'controller_oracle_max_error':oracle_error,'effective_delta_roundtrip_max_error':maximum_error,
               'raw_clipping_max_difference':max_clip,'rot6d_roundtrip_error':rot_error,
               'base_mode_frames':int(np.sum(actions[:,4]>0)),'auxiliary_controls_preserved':True}
        entries.append(entry);print(json.dumps(entry),flush=True)
    env.close()
    return {'benchmark':'RoboCasa365','source':str(args.source),'episodes':entries,
            'frame':'robot base center; additional world targets recorded separately','body_to_tool_rotation':body_to_tool.tolist(),
            'live_orientation_calibration_max_error':calibration_error,'live_position_calibration_max_error':position_error,
            'controller_source':str(controller_path),'controller_sha256':digest(controller_path),
            'action_scale':scales.tolist(),'mode_handling':'mode<=0: achieved observed tool pose; mode>0: previous desired target',
            'validation':'All rows checked against native OSC target functions and inverse action conversion; no full dynamics replay',
            'important':'Auxiliary base/mode labels remain separate. Stock X-VLA EE6D handler cannot train mobile controls; do not discard or reinterpret padding.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['libero','robocasa'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    record=prepare_libero(args) if args.benchmark=='libero' else prepare_robocasa(args)
    record.update(status='converted_and_validated',total_frames=sum(r['frames'] for r in record['episodes']),
                  converter_sha256=digest(__file__),schema='xvla-aligned-absolute-v1',training_performed=False)
    (args.output/'conversion_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('CONVERSION_COMPLETE',args.benchmark,record['total_frames'],flush=True)


if __name__=='__main__':main()
