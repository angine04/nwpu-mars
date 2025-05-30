import sys
from engine.engine import MarsEngine


if __name__ == "__main__":
    mode = "pipe"
    nobuf = False

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"

    MarsEngine(
        mode=mode,
        cfgname="vanilla.nano.full.pretrained",
        root="/home/v5/Mars", # 注意项目运行root不要放在代码路径下
        nobuf=nobuf,
    ).run()
