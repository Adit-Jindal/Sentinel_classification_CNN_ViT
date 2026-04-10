#!/bin/bash

MODE=$1

echo "Running mode: $MODE"

# -----------------------------
# TRAIN
# -----------------------------
run_all_train() {
    time python train11.py
    time python train12.py
    time python train21.py
    time python train22.py
}

# -----------------------------
# TEST
# -----------------------------
run_all_test() {
    time python test11.py
    time python test12.py
    time python test21.py
    time python test22.py
}

# -----------------------------
# ABLATION
# -----------------------------
run_all_ablation() {
    time python ablation11.py
    time python ablation12.py
    time python ablation21.py
    time python ablation22.py
}

# -----------------------------
# PER TASK BLOCKS
# -----------------------------
run11() {
    time python train11.py
    time python test11.py
    time python ablation11.py
}

run12() {
    time python train12.py
    time python test12.py
    time python ablation12.py
}

run21() {
    time python train21.py
    time python test21.py
    time python ablation21.py
}

run22() {
    time python train22.py
    time python test22.py
    time python ablation22.py
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