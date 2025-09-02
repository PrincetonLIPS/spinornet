# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Spinornet authors.
# All Spinornet modifications are Copyright 2025 Spinornet authors.

import datetime
import os
from typing import Optional
from absl import logging
import numpy as np
import pickle
import jax
import dataclasses
import jax.numpy as jnp


def create_save_path(save_path: Optional[str]) -> str:
    """Creates the directory for saving checkpoints, if it doesn't exist.

    Args:
      save_path: directory to use. If false, create a directory in the working
        directory based upon the current time.

    Returns:
      Path to save checkpoints to.
    """
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    default_save_path = os.path.join(os.getcwd(), f'spinornet_{timestamp}')
    ckpt_save_path = save_path or default_save_path
    if ckpt_save_path and not os.path.isdir(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    return ckpt_save_path


def save_params(ckpt_dir: str, state, t: int):
    logging.info('Saving checkpoint %s', f'{t:06d}')
    with open(os.path.join(ckpt_dir, f"arrays-{t:06d}.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, f"tree-{t:06d}.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore_params(ckpt_dir, t: int):
    with open(os.path.join(ckpt_dir, f"tree-{t:06d}.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, f"arrays-{t:06d}.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]
    return jax.tree_unflatten(treedef, flat_state)


def save(save_path: str, t: int, data, feats, opt_state, mcmc_width, mcmc_spin_width, pmove, stats) -> str:
    """Saves checkpoint information to a npz file.

    Args:
      save_path: path to directory to save checkpoint to. The checkpoint file is
        save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
        iterations.
      t: number of completed iterations.
      data: MCMC walker configurations.
      feats: MCMC walker spins.
      opt_state: optimization state.
      mcmc_width: width to use in the MCMC proposal distribution.
      mcmc_spin_width: width to use in the MCMC spin proposal distribution.
      pmove: probability of moving a walker.
      stats: auxiliary statistics.

    Returns:
      path to checkpoint file.
    """
    ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
    logging.info('Saving checkpoint %s', ckpt_filename)
    with open(ckpt_filename, 'wb') as f:
        np.savez(
            f,
            t=t,
            data=data,
            feats=feats,
            opt_state=opt_state,
            mcmc_width=mcmc_width,
            mcmc_spin_width=mcmc_spin_width,
            pmove=pmove,
            stats=stats,

        )
    return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None, shape_check=True):
    """Restores data saved in a checkpoint.

    Args:
      restore_filename: filename containing checkpoint.
      batch_size: total batch size to be used. If present, check the data saved in
        the checkpoint is consistent with the batch size requested for the
        calculation.

    Returns:
      (t, data, feats, opt_state, mcmc_width, mcmc_spin_width, pmove, stats) tuple, where
      t: number of completed iterations.
      data: MCMC walker configurations.
      feats: MCMC walker spins.
      opt_state: optimization state.
      mcmc_width: width to use in the MCMC proposal distribution.
      mcmc_spin_width: width to use in the MCMC spin proposal distribution.
      pmove: probability of moving a walker.
      stats: auxiliary statistics.

    Raises:
      ValueError: if the leading dimension of data does not match the number of
      devices (i.e. the number of devices being parallelised over has changed) or
      if the total batch size is not equal to the number of MCMC configurations in
      data.
    """
    logging.info('Loading checkpoint %s', restore_filename)
    with open(restore_filename, 'rb') as f:
        ckpt_data = np.load(f, allow_pickle=True)
        # Retrieve data from npz file. Non-array variables need to be converted back
        # to natives types using .tolist().
        t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
        data = ckpt_data['data']
        feats = ckpt_data['feats']
        opt_state = ckpt_data['opt_state'].tolist()
        mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())
        mcmc_spin_width = jnp.array(ckpt_data['mcmc_spin_width'].tolist())
        pmove = jnp.array(ckpt_data['pmove'].tolist())
        stats = ckpt_data['stats']
        if shape_check:
            if data.shape[0] != jax.local_device_count():
                raise ValueError(
                    'Incorrect number of devices found. Expected {}, found {}.'.format(
                        data.shape[0], jax.local_device_count()))
            if batch_size and data.shape[0] * data.shape[1] != batch_size:
                raise ValueError(
                    'Wrong batch size in loaded data. Expected {}, found {}.'.format(
                        batch_size, data.shape[0] * data.shape[1]))
    return t, data, feats, opt_state, mcmc_width, mcmc_spin_width, pmove, stats
