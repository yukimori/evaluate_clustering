# -*- coding:utf-8 -*-

import os
import configparser
import json

class Config:
    COMMON_PARAM_STARTS_WITH= "$$"
    
    def __init__(self, filepath='conf/clustering.ini', home='./'):
        self.home = home
        self.inifile = configparser.SafeConfigParser()
        if filepath is not None and os.path.isfile(filepath):
            self.configfile = filepath
            self.inifile.read(self.configfile)
            print("{0} loaded.".format(self.configfile))
        else:
            print("{0} can't load.".format(filepath))
            import sys
            sys.exit(1)
        self.common_settings = {}
        self.sub_configs = {}
        self.read_common_settings()

    # 共通設定を読み込む
    def read_common_settings(self):
        self.common_settings.update(self.get_items('common'))
        # print(self.common_settings)

    # 指定したサブ設定を共通設定に変換する
    def replace_sub_config(self, **sub_config):
        for key in sub_config.keys():
            if isinstance(sub_config[key], dict):
                sub_config[key] = self.replace_sub_config(**sub_config[key])
            elif isinstance(sub_config[key], str):
                if sub_config[key].startswith(Config.COMMON_PARAM_STARTS_WITH):
                    common_key = (sub_config[key])[2:]
                    # print(self.common_settings)
                    # print(self.common_settings[common_key])
                    value, ttype = self.common_settings[common_key].split(",")
                    if ttype == "int":
                        sub_config[key] = int(value)
                    else:
                        raise Exception(message="unsupported type")
        return sub_config


    # サブ設定を取得する
    def get_sub_config(self, sub_config_name):
        if sub_config_name in self.sub_configs:
            return self.sub_configs[sub_config_name]
        with open(self.get_config('conf', sub_config_name)) as f:
            sub_config = json.load(f)
            sub_config = self.replace_sub_config(**sub_config)
            self.sub_configs[sub_config_name] = sub_config

            return sub_config

    def get_config(self, section, key):
        return os.path.join(self.home, self.inifile.get(section, key))

    def get_items(self, section):
        return dict(self.inifile.items(section))
