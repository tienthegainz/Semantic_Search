"""
    FCam.database
    -------

    This module implement database of FCam.

    :copyright 2019 by FTECH team.
"""
import pickle
import numpy as np

import config

class DBManagement():
    __instance = None
    global data

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DBManagement.__instance == None:
            DBManagement()
        return DBManagement.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if DBManagement.__instance != None:
            print ('DB class is a singleton!')
        else:
            DBManagement.__instance = self
            self.get_database(config.db_file)

    def get_database(self, db_file):
        try:
            self.data = pickle.loads(open(db_file, "rb").read())
        except:
            return

    def save(self):
        f = open(config.db_file, "wb")
        f.write(pickle.dumps(self.data))
        f.close()

    def save_data(self,features, genders, ages, clusters):
        f = open(config.db_file, "wb")
        self.data = {"features": features, "genders": genders, "ages": ages, "clusters": clusters}
        f.write(pickle.dumps(self.data))
        f.close()
