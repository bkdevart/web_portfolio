from models import app, GameLog, GameAttr
from sqlalchemy.sql import select
import pandas as pd

@app.route('/log/', methods=['POST'])
def pull_log():
    '''
    returns data for graphing
    '''
    print('testing log...')