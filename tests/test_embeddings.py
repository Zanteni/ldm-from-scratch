import torch
from models.embeddings import SinusoidalTimeEmbedding, TimestepMLP


def test_sinusoidal_shape():
    dim = 512
    emb = SinusoidalTimeEmbedding(dim=dim)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    assert out.shape == (4, dim), f"Expected (4,{dim}), got {out.shape}"
    print(f"[OK] sinusoidal shape test passed: {tuple(out.shape)}")


def test_sinusoidal_different_t_gives_different_embeddings():
    dim = 512
    emb = SinusoidalTimeEmbedding(dim=dim)
    t = torch.tensor([0, 500])
    out = emb(t)
    assert not torch.allclose(out[0], out[1]), \
        "Different timesteps produced identical embeddings."
    print("[OK] different t -> different embeddings test passed")


def test_sinusoidal_nearby_t_more_similar_than_distant_t():
    """Correctness check: t=0 and t=1 should be much more similar to each
    other than t=0 and t=999 -- confirms the smooth "closeness" property
    sinusoidal embeddings are supposed to provide."""
    dim = 512
    emb = SinusoidalTimeEmbedding(dim=dim)
    t = torch.tensor([0, 1, 999])
    out = emb(t)

    dist_near = (out[0] - out[1]).pow(2).sum().item()
    dist_far = (out[0] - out[2]).pow(2).sum().item()

    assert dist_near < dist_far, \
        f"Expected nearby t to be more similar than distant t, got dist_near={dist_near:.4f}, dist_far={dist_far:.4f}"
    print(f"[OK] nearby-t-more-similar test passed: dist(t=0,t=1)={dist_near:.4f} < dist(t=0,t=999)={dist_far:.4f}")


def test_timestep_mlp_shape():
    mlp = TimestepMLP(embedding_dim=128, hidden_dim=512, out_dim=512)
    t = torch.randint(0, 1000, (4,))
    out = mlp(t)
    assert out.shape == (4, 512), f"Expected (4,512), got {out.shape}"
    print(f"[OK] TimestepMLP shape test passed: {tuple(out.shape)}")


def test_timestep_mlp_gradient_flow():
    mlp = TimestepMLP(embedding_dim=128, hidden_dim=512, out_dim=512)
    t = torch.randint(0, 1000, (4,))
    out = mlp(t)
    loss = out.mean()
    loss.backward()

    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] TimestepMLP gradient flow test passed")


def run_embeddings_suite():
    print("===== EMBEDDINGS TEST =====")
    test_sinusoidal_shape()
    test_sinusoidal_different_t_gives_different_embeddings()
    test_sinusoidal_nearby_t_more_similar_than_distant_t()
    test_timestep_mlp_shape()
    test_timestep_mlp_gradient_flow()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_embeddings_suite()