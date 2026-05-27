"""morph_gs — MORPH Foundation Model adapted for Grad-Shafranov warm-start.

Quickstart::

    from morph_gs import MorphGS, GSDatasetV2, FieldStats, cold_solve, warm_solve

    # Training
    train_ds = GSDatasetV2("data/gs_dataset_v2.h5", split="train")
    val_ds   = GSDatasetV2("data/gs_dataset_v2.h5", split="val",
                           field_stats=train_ds.stats)
    model = MorphGS("models/morph-Ti-FM-max_ar1_ep225.pth",
                    ft_level=1, lora_r_attn=16, lora_r_mlp=16, lora_alpha=32)
    optimizer = model.configure_optimizer(lr_head=1e-3, lr_backbone=5e-4)
    # ... standard PyTorch training loop ...

    # Warm-start evaluation
    test_ds = GSDatasetV2("data/gs_dataset_v2.h5", split="test",
                          field_stats=train_ds.stats)
    raw = test_ds.get_solver_inputs(idx=0)
    iters_cold, _, _ = cold_solve(raw)
    psi_pred = predict_psi(model, test_ds, idx=0)
    iters_warm, _, _ = warm_solve(raw, psi_init_pred=psi_pred)
"""

def __getattr__(name):
    # Lazy imports so that `python -m morph_gs.process_shot` (workers) does
    # not load torch/model unless explicitly requested.
    _map = {
        "MorphGS":            ("morph_gs.model",      "MorphGS"),
        "MorphGSE":           ("morph_gs.model",      "MorphGSE"),
        "UpsamplingDecoder":  ("morph_gs.model",      "UpsamplingDecoder"),
        "BilinearDecoder":    ("morph_gs.model",      "BilinearDecoder"),
        "GSDatasetV2":        ("morph_gs.dataset_v2", "GSDatasetV2"),
        "FieldStats":         ("morph_gs.dataset_v2", "FieldStats"),
        "build_input_fields": ("morph_gs.fields",     "build_input_fields"),
        "compute_psi_init":   ("morph_gs.fields",     "compute_psi_init"),
        "build_machine":      ("morph_gs.fields",     "build_machine"),
        "cold_solve":         ("morph_gs.solver",     "cold_solve"),
        "warm_solve":         ("morph_gs.solver",     "warm_solve"),
        "predict_psi":        ("morph_gs.solver",     "predict_psi"),
        "process_shot":       ("morph_gs.process_shot", "process_shot"),
        "build_dataset":      ("morph_gs.build_dataset", "build_dataset"),
        "train":              ("morph_gs.train",      "train"),
        "eval_pairs":         ("morph_gs.eval",       "eval_pairs"),
    }
    if name in _map:
        import importlib
        mod_name, attr = _map[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'morph_gs' has no attribute {name!r}")


__all__ = [
    "MorphGS", "MorphGSE", "UpsamplingDecoder", "BilinearDecoder",
    "GSDatasetV2", "FieldStats",
    "build_input_fields", "compute_psi_init", "build_machine",
    "cold_solve", "warm_solve", "predict_psi",
    "process_shot", "build_dataset", "train", "eval_pairs",
]
