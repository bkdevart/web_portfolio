import os
from datetime import datetime, date, timedelta
from math import pi
from flask import Flask, render_template
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.palettes import RdBu
from bokeh.transform import cumsum
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gameplay/')
def gameplay():
    plots = []
    # grab source data
    source, complete, game_log, completed = init_data()

    # add weekly hours for most played game plot
    plot = game_of_the_week(source)
    plots.append(components(plot))

    # add weekly hours distribution pie graph
    plot = weekly_hours_snapshot(source)
    plots.append(components(plot))

    # add top 10 streaks graph
    plot = check_streaks(source)
    plots.append(components(plot))

    # add weekly hours graph
    plot = line_weekly_hours(source)
    plots.append(components(plot))

    # add rank by game time graph
    plot = bar_graph_top(source)
    plots.append(components(plot))

    # add top game time pie graph
    plot = pie_graph_top(source)
    plots.append(components(plot))

    return render_template('gameplay.html', plots=plots,
                           game_log=game_log, completed=completed)


def init_data():
    '''
    Initializes data, returning two dataframes - source and complete
    '''
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    game_log = os.path.join(THIS_FOLDER, 'input/game_log.csv')
    completed = os.path.join(THIS_FOLDER, 'input/completed.csv')
    source = pd.read_csv(game_log,
                         parse_dates=['date', 'time_played'])
    complete = pd.read_csv(completed,
                           parse_dates=['date_completed'])
    # storing initial csv states for how-to section
    game_log = source.head(3).to_html(index=False)
    completed = complete.head(3).to_html(index=False)

    # perform initial calculations
    source['minutes_played'] = ((source['time_played'].dt.hour * 60)
                                + source['time_played'].dt.minute)
    source['hours_played'] = source['minutes_played'] / 60
    source['dow'] = source['date'].dt.weekday_name
    source.sort_values('date', inplace=True)
    source['cum_total_minutes'] = (source.groupby('title')
                                   ['minutes_played'].cumsum())
    # add in start week (Sunday - Sat)
    # subtract the day number from the date (with 0 being Monday)
    source['week_start'] = (source['date'] - pd.to_timedelta
                            (source['date'].dt.dayofweek, unit='d'))
    source['month'] = source['date'].values.astype('datetime64[M]')

    return source, complete, game_log, completed


def game_of_the_week(source_data, num_weeks=16):
    '''
    Shows each week's longest-played game and the hours spent playing for
    that week

    Parameters
    ----------
    num_weeks : int
        This determines the number of weeks to display, beginning from the
        current week and counting backwards.  Defaults to 16.

    Returns
    -------
    weekly_top_games : DataFrame
        The contents DataFrame are displayed graphically.  Fields below.

        week_start : datetime64[ns]
            week-start date (Monday-based) for top game of that week
        hours_played : float64
            Total hours played per game, per week
        title : object (str)
            Name of game played
    '''
    weekly_game_hours = (source_data.groupby(['week_start', 'title'],
                                             as_index=False)
                         [['hours_played']].sum())
    weekly_top_hours = (weekly_game_hours.groupby(['week_start'],
                                                  as_index=False).max()
                        [['week_start', 'hours_played']])
    # TODO: see if there's a more efficient way of doing this
    weekly_top_games = (weekly_top_hours
                        .merge(weekly_game_hours
                               [['week_start', 'hours_played', 'title']],
                               on=['week_start', 'hours_played'],
                               how='left'))

    graph = weekly_top_games[['title', 'hours_played']].tail(num_weeks)
    # plot graph df with bokeh
    source = ColumnDataSource(graph)
    y_range = list(set(graph['title']))

    plot = figure(plot_height=300,
                  sizing_mode='scale_width',
                  y_range=y_range,
                  title='Hours Per Week')
    plot.hbar(y='title',
              source=source,
              right='hours_played',
              height=.5,
              line_color='#8e8d7d',
              fill_color='#8e8d7d')
    most_recent_week = weekly_top_games['week_start'].dt.date.iloc[-1]
    curr_top_game = weekly_top_games['title'].iloc[-1]
    # TODO: convert this line to a string, figure out how display in HTML
    # top_game = f'Top Game for week of {most_recent_week}: {curr_top_game}'
    return plot


def weekly_hours_snapshot(source):
    '''
    Displays pie chart of time played per game in curretn week
    '''
    df = __weekly_hours_by_game(source)
    # filter to current week
    current_week = df['week_start'].max()
    df = (df[df['week_start'] == current_week]
          [['title', 'hours_played']])
    # some calculations to orientate the pie wedges
    df['angle'] = (df['hours_played']/
                   df['hours_played'].sum() * 2*pi)
    df['color'] = RdBu[len(df['hours_played'])]
    source = ColumnDataSource(df)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title='Weekly Hours Distribution')
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='title',
            fill_color='color', source=source)
    return p


def __weekly_hours_by_game(source):
    '''
    Shows total hours played for each game for each week

    Returns
    -------
    weekly_game_hours : DataFrame
        week_start : datetime64[ns]
            Week-start date (Monday-based) for title played
        title : object (str)
            Title of game
        hours_played : float64[ns]
            Total hours spent playing title for the week
    '''
    weekly_game_hours = (source.groupby(['week_start',
                                         'title'],
                                        as_index=False)
                         [['hours_played']].sum().sort_values(
                                                  ['week_start',
                                                   'hours_played'],
                                                  ascending=False))
    return weekly_game_hours


def check_streaks(source, top_games=10):
    '''
    Evaluates which games have been played at least two consecutive days
    in a row.  Prints any games that are currently being played in a streak
    and graphs the longest streaks along with the days for each
    title.

    Parameters
    ----------
    top_games : int
        Number of games to display in graph.  Graph starts from the game
        with the most consecutive days.

    Returns
    -------
    max_streak : DataFrame
        Superset of data used in graph displaying all games that have been
        played in a streak, along with the number of consecutive days.
        Fields below.

        index : object (str)
            Date game was played
        days : float64
            Days played in a continuous streak

    '''
    df = get_streaks(source)
    # calculate current streak (streaks with yesterday's date)
    # filter down to results with yesterday's date
    yesterday = pd.Timestamp(date.today() - timedelta(1))
    current_streak = (df[(df['next_day'] == yesterday)][['title',
                                                         'streak_num']])
    # turn the title(s) into a list to print
    # if current_streak empty, put 'None' in list
    # TODO: adapt this to string and find out how to display
    '''
    print('\nCurrent streak(s):')
    if len(current_streak) == 0:
        print('None')
    else:
        current_streak['days'] = current_streak['streak_num'] + 1
        current_streak = (current_streak[['title', 'days']]
                          .set_index('title'))
        print(current_streak)
    '''
    # take max streak_sum, grouped by title, store in new object
    max_streak = df.groupby('title', as_index=False)['streak_num'].max()
    max_streak['days'] = max_streak['streak_num'] + 1
    max_streak = max_streak[['title', 'days']].sort_values(['days'],
                                                           ascending=False)
    # graph data
    # adapt this for bokeh, return plot
    graph = max_streak[['title', 'days']].head(top_games)
    # plot graph df with bokeh
    source = ColumnDataSource(graph)
    y_range = list(set(graph['title']))
    title = 'Top ' + str(top_games) + ' Streaks'

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               y_range=y_range,
               title=title)
    p.hbar(y='title',
           source=source,
           right='days',
           height=.5,
           line_color='#8e8d7d',
           fill_color='#8e8d7d')
    return p


def get_streaks(source):
    # data needed: game title and date
    df = source[['title', 'date']]
    # order by game title first, then date
    df = df.sort_values(['title', 'date'])
    # use groupby with title, shift date down by one to get next day played
    df['next_day'] = df.groupby('title')['date'].shift(-1)
    # fill nat values in next_day with date values (for subtraction)
    df['next_day'] = df['next_day'].fillna(df['date'])
    # subtract number of days between date and next_day, store in column
    df['consecutive'] = (df['next_day'] - df['date'])
    # if column value = 1 day, streak = true (new column)
    df = df[(df['consecutive'] == timedelta(days=1))]
    # need to group streaks
    # test: date - date.shift(-1) = 1 day, and same for next_day
    df['group'] = (((df['next_day'] - df['next_day'].shift(1))
                    == timedelta(days=1)) &
                   ((df['date'] - df['date'].shift(1))
                    == timedelta(days=1)))
    # false represents the beginning of each streak, so it equals 1
    df['streak_num'] = np.where(df['group'] == False, 1, np.nan)
    # forward fill streak_num to complete streak count
    for col in df.columns:
        g = df['streak_num'].notnull().cumsum()
        df['streak_num'] = (df['streak_num'].fillna(method='ffill') +
                            df['streak_num'].groupby(g).cumcount())
    return df


def line_weekly_hours(source):
    '''
    Graphs total weekly hours on a line graph
    '''
    graph = __agg_week(source)[['week_start', 'hours_played',
                                'avg_hrs_per_day']]
    graph = graph.set_index('week_start')
    # add this as bokeh graph, return plot
    title = 'Weekly Hours'
    num_lines = len(graph.columns)
    my_palette = ['#8e8d7d', '#acd1e0']

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title,
               x_axis_type='datetime')
    p.multi_line(xs=[graph.index.values]*num_lines,
                 ys=[graph[name].values for name in graph],
                 line_color=my_palette,
                 line_width=2)
    return p

def __agg_week(source):
    '''
    Creates a DataFame that rolls up weekly totals.  Does not include game
    title information.

    Returns
    -------
    weekly_hour_days : DataFrame
        week_start : datetime64[ns]
            The date of the first Sunday of each week
        hours_played : float64
            Total hours played for the week
        days_sampled : int64
            Number of days containing data for the week
        avg_hrs_per_day : float64
            hours_played / days_sampled
    '''
    weekly_hour = (source.groupby('week_start', as_index=False)
                   .sum()[['week_start', 'hours_played']])
    weekly_days = (source.groupby('week_start').nunique()
                   [['date']].reset_index())
    weekly_hour_days = pd.merge(weekly_hour, weekly_days, on='week_start')
    weekly_hour_days.columns = ['week_start',
                                'hours_played',
                                'days_sampled']
    weekly_hour_days['avg_hrs_per_day'] = (weekly_hour_days['hours_played']
                                           /
                                           weekly_hour_days['days_sampled']
                                           )
    return weekly_hour_days


def bar_graph_top(source, num_games=10):
    '''
    This function will create a horizontal bar chart representing total
    time spent playing individual games.
        - Time is ranked with longest at the bottom

    Parameters
    ----------
    num_games : int
        This determines the number of games to display, starting from the
        top.  Defaults to 10.
    '''
    # set data
    graph = __current_top(__agg_total_time_played(source), num_games)

    n_groups = graph['title'].count()
    game_rank = graph['hours_played']
    tick_names = graph['title']

    # add bokeh instructions, return plot
    source = ColumnDataSource(graph)
    y_range = list(set(graph['title']))
    title = 'Rank by Game Time'

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               y_range=y_range,
               title=title)
    p.hbar(y='title',
           source=source,
           right='hours_played',
           height=.5,
           line_color='#8e8d7d',
           fill_color='#8e8d7d')
    return p
    '''
    # create plots
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.barh(index + bar_width, game_rank, bar_width,
             alpha=opacity,
             color='g',
             label='Game')

    plt.ylabel('Game')
    plt.xlabel('Hours Played')
    plt.title('Rank by Game Time')
    plt.yticks(index + bar_width, tick_names)

    plt.tight_layout()
    plt.savefig((self._config['output'] + 'games.png'),
                dpi=300)
    plt.show()
    '''


def __agg_total_time_played(source):
    '''
    Sums up total time played by game, and ranks totals

    Returns
    -------
    time_played : DataFrame
        title : object (str)
            Title of game
        minutes_played : int64
            Total minutes spent playing the game since data tracking began
        hours_played : float64
            Total hours spent playing the game since data tracking began
        rank : float64
            Rank of game based on time spent playing, with 1 representing
            the most time spent
    '''
    time_played = pd.DataFrame(source.groupby('title',
                                              as_index=False)
                               [['title',
                                 'minutes_played',
                                 'hours_played']].sum())
    time_played.sort_values('minutes_played', ascending=False,
                            inplace=True)
    time_played['rank'] = (time_played['minutes_played']
                           .rank(method='dense', ascending=False))
    return time_played


def __current_top(source_df, rank_num):
    '''
    This generates data for the specified number of top ranked games
        - Primarily used to limit data points in graphs

    Returns
    -------
    top : DataFrame
        title : object (str)
            Title of game
        minutes_played : int64
            Total minutes spent playing the game since data tracking began
        hours_played : float64
            Total hours spent playing the game since data tracking began
        rank : float64
            Overall rank by time played
    '''
    top = source_df[source_df['rank'] <= rank_num]
    return top


def pie_graph_top(source, num_games=10):
    '''
    This creates a pie plot of the number of games specified
        - Focuses on overall time spent

    Parameters
    ----------
    num_games : int
        This determines the number of games to display, starting from the
        top.  Defaults to 10.
    '''
    # pop out the first slice (assuming 10 items)
    # explode = (.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # plt.pie(game_rank, labels=tick_names, explode=explode)
    graph = __current_top(__agg_total_time_played(source), num_games)
    game_rank = graph['hours_played']
    tick_names = graph['title']
    title = 'Top ' + str(num_games) + ' Distro'
    # add graph using bokeh, return plot
    graph['angle'] = (graph['hours_played']/
                      graph['hours_played'].sum() * 2*pi)
    graph['color'] = RdBu[len(graph['hours_played'])]
    source = ColumnDataSource(graph)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title)
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='title',
            fill_color='color', source=source)
    return p


@app.route('/music/')
def music():
    return render_template('music.html')


@app.route('/exercise/')
def exercise():
    return render_template('exercise.html')
