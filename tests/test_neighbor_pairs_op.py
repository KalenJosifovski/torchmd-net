# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pytest
import torch
import torchmdnet.extensions.torchmdnet_extensions  # noqa: F401


def _reference_neighbor_pairs(
    positions,
    batch,
    cutoff_lower,
    cutoff_upper,
    loop=False,
    include_transpose=False,
):
    n_atoms = positions.size(0)
    neighbors = torch.tril_indices(n_atoms, n_atoms, -1, device=positions.device)
    mask = batch[neighbors[0]] == batch[neighbors[1]]
    neighbors = neighbors[:, mask].to(torch.int32)
    deltas = positions[neighbors[0]] - positions[neighbors[1]]
    distances = torch.linalg.vector_norm(deltas, dim=1)
    mask = (distances < cutoff_upper) & (distances >= cutoff_lower)
    neighbors = neighbors[:, mask]
    deltas = deltas[mask]
    distances = distances[mask]
    if include_transpose:
        neighbors = torch.hstack([neighbors, torch.stack([neighbors[1], neighbors[0]])])
        distances = torch.hstack([distances, distances])
        deltas = torch.vstack([deltas, -deltas])
    if loop:
        rng = torch.arange(0, n_atoms, dtype=torch.int32, device=positions.device)
        neighbors = torch.hstack([neighbors, torch.stack([rng, rng])])
        distances = torch.hstack([distances, torch.zeros_like(rng, dtype=distances.dtype)])
        deltas = torch.vstack(
            [deltas, torch.zeros((n_atoms, 3), dtype=deltas.dtype, device=positions.device)]
        )
    return neighbors, deltas, distances


def _sorted_outputs(edge_index, edge_vec, edge_weight, num_atoms):
    mask = edge_index[0] != -1
    edge_index = edge_index[:, mask].to(torch.long)
    edge_vec = edge_vec[mask]
    edge_weight = edge_weight[mask]
    keys = edge_index[0] * num_atoms + edge_index[1]
    order = torch.argsort(keys)
    return edge_index[:, order], edge_vec[order], edge_weight[order]


def _run_neighbor_pairs(device):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    batch = torch.zeros(positions.size(0), dtype=torch.long, device=device)
    box = torch.eye(3, device=device)
    cutoff_lower = 0.0
    cutoff_upper = 1.6
    max_num_pairs = 16
    edge_index, edge_vec, edge_weight, num_pairs = (
        torch.ops.torchmdnet_extensions.get_neighbor_pairs(
            "brute",
            positions,
            batch,
            box,
            False,
            cutoff_lower,
            cutoff_upper,
            max_num_pairs,
            False,
            False,
        )
    )
    assert edge_index.shape == (2, max_num_pairs)
    assert edge_vec.shape == (max_num_pairs, 3)
    assert edge_weight.shape == (max_num_pairs,)
    assert num_pairs.numel() == 1
    return positions, batch, edge_index, edge_vec, edge_weight


def test_get_neighbor_pairs_cpu_dispatch_and_output():
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "torchmdnet_extensions::get_neighbor_pairs", "CPU"
    )
    positions, batch, edge_index, edge_vec, edge_weight = _run_neighbor_pairs("cpu")
    ref_index, ref_vec, ref_weight = _reference_neighbor_pairs(
        positions,
        batch,
        cutoff_lower=0.0,
        cutoff_upper=1.6,
        loop=False,
        include_transpose=False,
    )
    edge_index, edge_vec, edge_weight = _sorted_outputs(
        edge_index, edge_vec, edge_weight, positions.size(0)
    )
    ref_index, ref_vec, ref_weight = _sorted_outputs(
        ref_index, ref_vec, ref_weight, positions.size(0)
    )
    torch.testing.assert_close(edge_index, ref_index)
    torch.testing.assert_close(edge_vec, ref_vec)
    torch.testing.assert_close(edge_weight, ref_weight)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)
def test_get_neighbor_pairs_mps_dispatch_and_output():
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "torchmdnet_extensions::get_neighbor_pairs", "MPS"
    )
    positions, batch, edge_index, edge_vec, edge_weight = _run_neighbor_pairs("mps")
    ref_index, ref_vec, ref_weight = _reference_neighbor_pairs(
        positions.cpu(),
        batch.cpu(),
        cutoff_lower=0.0,
        cutoff_upper=1.6,
        loop=False,
        include_transpose=False,
    )
    edge_index, edge_vec, edge_weight = _sorted_outputs(
        edge_index.cpu(), edge_vec.cpu(), edge_weight.cpu(), positions.size(0)
    )
    ref_index, ref_vec, ref_weight = _sorted_outputs(
        ref_index, ref_vec, ref_weight, positions.size(0)
    )
    torch.testing.assert_close(edge_index, ref_index)
    torch.testing.assert_close(edge_vec, ref_vec)
    torch.testing.assert_close(edge_weight, ref_weight)
