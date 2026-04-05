#!/bin/bash

MODE=$1

echo "Running mode: $MODE"

# -----------------------------
# TRAIN
# -----------------------------
run_all_train() {
    python train11.py
    python train12.py
    python train21.py
    python train22.py
}

# -----------------------------
# TEST
# -----------------------------
run_all_test() {
    python test11.py
    python test12.py
    python test21.py
    python test22.py
}

# -----------------------------
# ABLATION
# -----------------------------
run_all_ablation() {
    python ablation11.py
    python ablation12.py
    python ablation21.py
    python ablation22.py
}

# -----------------------------
# PER TASK BLOCKS
# -----------------------------
run11() {
    python train11.py
    python test11.py
    python ablation11.py
}

run12() {
    python train12.py
    python test12.py
    python ablation12.py
}

run21() {
    python train21.py
    python test21.py
    python ablation21.py
}

run22() {
    python train22.py
    python test22.py
    python ablation22.py
}

# -----------------------------
# EXECUTION MODES
# -----------------------------
case $MODE in

    # 1. all training
    train_all)
        run_all_train
        ;;

    # 2. all testing
    test_all)
        run_all_test
        ;;

    # 3. all ablation
    ablation_all)
        run_all_ablation
        ;;

    # 4. task 1.1
    11)
        run11
        ;;

    # 5. task 1.2
    12)
        run12
        ;;

    # 6. task 2.1
    21)
        run21
        ;;

    # 7. task 2.2
    22)
        run22
        ;;

    # 8. full ordered pipeline
    full)
        run11
        run12
        run21
        run22
        ;;

    *)
        echo "Invalid mode."
        echo "Usage:"
        echo "./run.sh train_all"
        echo "./run.sh test_all"
        echo "./run.sh ablation_all"
        echo "./run.sh 11 | 12 | 21 | 22"
        echo "./run.sh full"
        ;;
esac

echo "Done."