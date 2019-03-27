import sys
sys.path.append('../')

from valuate import *


def process_all_models():
    """
    执行所有车型流程
    """
    process = Process()
    process.process_all_models()


def process_cron():
    """
    执行日常
    """
    process = Process()
    process.process_cron()


if __name__ == "__main__":
    # process_all_models()
    process_cron()