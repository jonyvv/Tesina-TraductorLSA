# -*- coding: utf-8 -*-
"""
Tests de la construccion de folds leave-one-subject-out.

No necesitan torch: validan la particion, que es donde puede aparecer fuga de
datos. El loop de entrenamiento se prueba corriendo ml/evaluate_loso.py.

    python ml/tests/test_lsa64_loso.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from lsa64.evaluation import construir_folds, verificar_fold

N_SUJETOS, N_CLASES, N_REPS = 10, 64, 5


def _subjects(n_sujetos=N_SUJETOS, n_clases=N_CLASES, n_reps=N_REPS) -> list[str]:
    return [
        f"sujeto_{s:02d}"
        for _ in range(n_clases)
        for s in range(1, n_sujetos + 1)
        for _ in range(n_reps)
    ]


def test_hay_un_fold_por_sujeto():
    folds = construir_folds(_subjects())
    assert len(folds) == N_SUJETOS
    testeados = [f.test_subject for f in folds]
    assert sorted(testeados) == sorted({s for s in _subjects()})
    assert len(set(testeados)) == N_SUJETOS, "cada sujeto debe ser test exactamente una vez"


def test_ningun_fold_tiene_fuga():
    subjects = _subjects()
    for fold in construir_folds(subjects):
        assert verificar_fold(fold, subjects) == [], f"fold {fold.test_subject}: {verificar_fold(fold, subjects)}"


def test_particion_es_completa_y_disjunta():
    subjects = _subjects()
    total = len(subjects)
    for fold in construir_folds(subjects):
        idx = fold.train_idx + fold.val_idx + fold.test_idx
        assert len(idx) == total, "los tres splits deben cubrir todo el dataset"
        assert len(set(idx)) == total, "ninguna muestra puede estar en dos splits"
        assert len(fold.train_subjects) == N_SUJETOS - 2  # 10 - 1 test - 1 val


def test_val_sale_de_los_de_entrenamiento_nunca_del_test():
    subjects = _subjects()
    for fold in construir_folds(subjects, val_subjects=2):
        assert fold.test_subject not in fold.val_subjects
        assert fold.test_subject not in fold.train_subjects
        assert not set(fold.val_subjects) & set(fold.train_subjects)
        assert len(fold.val_subjects) == 2


def test_sin_validacion():
    for fold in construir_folds(_subjects(), val_subjects=0):
        assert fold.val_idx == []
        assert len(fold.train_subjects) == N_SUJETOS - 1


def test_es_determinista():
    a = construir_folds(_subjects())
    b = construir_folds(_subjects())
    assert [(f.test_subject, f.val_subjects, f.train_subjects) for f in a] == \
           [(f.test_subject, f.val_subjects, f.train_subjects) for f in b]


def test_falla_con_pocos_sujetos():
    for subjects in ([], ["sujeto_01"] * 5):
        try:
            construir_folds(subjects)
        except ValueError:
            continue
        raise AssertionError(f"deberia fallar con {len(set(subjects))} sujetos")


def test_detecta_fuga_inyectada():
    """Si alguien rompe construir_folds, verificar_fold tiene que avisar."""
    subjects = _subjects()
    fold = construir_folds(subjects)[0]
    fold.train_idx = fold.train_idx + fold.test_idx[:1]  # contamina train con test
    problemas = verificar_fold(fold, subjects)
    assert problemas and "fuga train/test" in problemas[0]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fallos = 0
    for test in tests:
        try:
            test()
            print(f"  OK   {test.__name__}")
        except AssertionError as exc:
            fallos += 1
            print(f"  FALLO {test.__name__}: {exc}")
        except Exception as exc:
            fallos += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - fallos}/{len(tests)} tests OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
