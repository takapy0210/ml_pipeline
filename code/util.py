import datetime
import logging
import sys,os
import numpy as np
import pandas as pd
import yaml
import joblib

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
SUB_DIR_NAME = yml['SETTING']['SUB_DIR_NAME']

# tensorflowとloggingのcollisionに対応
try:
    import absl.logging
    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def dump_df_pickle(cls, df, path):
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)


class Logger:

    def __init__(self, path):
        self.general_logger = logging.getLogger(path + 'general')
        self.result_logger = logging.getLogger(path + 'result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(path + 'general.log')
        file_result_handler = logging.FileHandler(path + 'result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name, path, sub_y_column):
        logger = Logger(path)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(RAW_DATA_DIR_NAME + 'sample_submission.csv')
        pred = Util.load_df_pickle(path + f'{run_name}-pred.pkl')
        submission[sub_y_column] = pred
        submission.to_csv(path + f'{run_name}_submission.csv', index=False)

        logger.info(f'{run_name} - end create submission')
