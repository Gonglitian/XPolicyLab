"""Synthetic LIBERO observations through the real policy server, without a simulator.

This still loads/executes the real checkpoint. It is not a model-free unit test.
"""
import argparse
import os

import numpy as np


def main():
    from client_server.ws import WsModelClient
    from .client import xpl_action_to_libero
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = 255
    color = image
    if os.environ.get('DEBUG_OBS_ENCODED') == '1':
        from XPolicyLab.utils.process_data import encode_image_bit
        color = encode_image_bit(image)
    obs = {'vision': {'agentview': {'color': color}, 'wrist': {'color': color}},
           'state': {'ee_pose': [0, 0, 0.3, 1, 0, 0, 0],
                     'gripper_qpos': [0.04, -0.04]}, 'instruction': 'pick up the bowl',
           'benchmark': 'libero', 'data_format_version': 'evomoe-benchmark-v1'}
    client = WsModelClient(url=f'ws://{args.host}:{args.port}', evaluation_id='libero-debug', trial_id='debug')
    try:
        client.call(func_name='reset')
        client.call(func_name='update_obs', obs=obs)
        chunk = client.call(func_name='get_action')
        if not chunk:
            raise ValueError('Empty action chunk')
        for action in chunk:
            if not np.isfinite(xpl_action_to_libero(action)).all():
                raise ValueError('Non-finite action')
        print(f'[DEBUG] received {len(chunk)} valid LIBERO actions; encoded={os.environ.get("DEBUG_OBS_ENCODED", "0")}')
    finally:
        client.close()


if __name__ == '__main__':
    main()
