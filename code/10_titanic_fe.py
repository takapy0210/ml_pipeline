import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
import warnings

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
Feature.dir = yml['SETTING']['FEATURE_PATH']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


# Target
class survived(Feature):
    def create_features(self):
        self.train['Survived'] = train['Survived']
        create_memo('Survived','生存フラグ。今回の目的変数。')


class passenger_id(Feature):
    def create_features(self):
        self.train['PassengerId'] = train['PassengerId']
        self.test['PassengerId'] = test['PassengerId']
        create_memo('PassengerId','搭乗者ID。')


class pclass(Feature):
    def create_features(self):
        self.train['Pclass'] = train['Pclass']
        self.test['Pclass'] = test['Pclass']
        create_memo('Pclass','チケットのクラス。1st, 2nd, 3rdの3種類')


class name(Feature):
    def create_features(self):
        self.train['Name'] = train['Name']
        self.test['Name'] = test['Name']
        create_memo('Name','名前')


class sex(Feature):
    def create_features(self):
        self.train['Sex'] = train['Sex']
        self.test['Sex'] = test['Sex']
        create_memo('Sex','性別')


class sex_label_encoder(Feature):
    def create_features(self):
        cols = 'Sex'
        tmp_df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
        le = LabelEncoder().fit(tmp_df[cols])
        self.train['sex_label_encoder'] = le.transform(train[cols])
        self.test['sex_label_encoder'] = le.transform(test[cols])
        create_memo('sex_label_encoder','性別をラベルエンコーディングしたもの')


class age(Feature):
    def create_features(self):
        self.train['Age'] = train['Age']
        self.test['Age'] = test['Age']
        create_memo('Age','年齢')


class age_mis_val_median(Feature):
    def create_features(self):
        self.train['Age_mis_val_median'] = train['Age'].fillna(train['Age'].median())
        self.test['Age_mis_val_median'] = test['Age'].fillna(test['Age'].median())
        create_memo('Age_mis_val_median','年齢の欠損値を中央値で補完したもの')


class sibsp(Feature):
    def create_features(self):
        self.train['SibSp'] = train['SibSp']
        self.test['SibSp'] = test['SibSp']
        create_memo('SibSp','兄弟/配偶者の数')


class parch(Feature):
    def create_features(self):
        self.train['Parch'] = train['Parch']
        self.test['Parch'] = test['Parch']
        create_memo('Parch','親/子供の数')


class family_size(Feature):
    def create_features(self):
        self.train['Family_Size'] = train['Parch'] + train['SibSp']
        self.test['Family_Size'] = test['Parch'] + test['SibSp']
        create_memo('Family_Size','家族の総数')


class ticket(Feature):
    def create_features(self):
        self.train['Ticket'] = train['Ticket']
        self.test['Ticket'] = test['Ticket']
        create_memo('Ticket','チケットナンバー')


class fare(Feature):
    def create_features(self):
        self.train['Fare'] = train['Fare']
        self.test['Fare'] = test['Fare']
        create_memo('Fare','運賃')


class fare_mis_val_median(Feature):
    def create_features(self):
        self.train['Fare_mis_val_median'] = train['Fare'].fillna(train['Fare'].median())
        self.test['Fare_mis_val_median'] = test['Fare'].fillna(test['Fare'].median())
        create_memo('Fare_mis_val_median','年齢の欠損値を中央値で補完したもの')


class cabin(Feature):
    def create_features(self):
        self.train['Cabin'] = train['Cabin']
        self.test['Cabin'] = test['Cabin']
        create_memo('Cabin','キャビン番号')


class embarked(Feature):
    def create_features(self):
        self.train['Embarked'] = train['Embarked']
        self.test['Embarked'] = test['Embarked']
        create_memo('Embarked','乗船した港')


# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '/_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path,"w"):pass

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])

if __name__ == '__main__':

    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)