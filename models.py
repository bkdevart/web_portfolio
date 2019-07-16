
from app import app

class GameLog(app.Model):
    '''
    Logs data by day and game title
    '''
    __tablename__ = 'game_log'
    title = app.Column(app.String(64))
    time_played = app.Column(app.String(10))
    date = app.Column(app.Date)
    description = app.Column(app.String(200))

class GameAttr(app.Model):
    '''
    Contains a list of game title meta data
    '''
    __tablename__ = 'game_attr'
    title = app.Column(app.String(64))
    complete = db.Column(app.Integer)
    date_completed = app.Column(app.Date)
    release_date = app.Column(app.Date)
    system = app.Column(app.String(20))
