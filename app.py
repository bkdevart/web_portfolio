# base imports
import os
from datetime import datetime, date, timedelta
from math import pi
from itertools import product
from flask import Flask, render_template, request

# graph imports
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.models.tools import HoverTool
from bokeh.embed import components
from bokeh.palettes import RdBu, Category20, cividis
from bokeh.transform import cumsum
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox, column
from bokeh.models.widgets import Slider

# analysis imports
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


@app.route('/art/')
def art():
    return render_template('art.html')


@app.route('/gameplay/')
def gameplay():
    # init dashboard element lists
    plots = []
    dynamic = []
    complete = []

    # init source data
    source, game_attr, game_log, game_attr_demo = init_game_data()
    source_sample = source.head(3).to_html(index=False)
    titles = (source[['title']]
              .drop_duplicates()
              .sort_values('title'))
    titles = titles['title'].tolist()
    current_title = request.args.get("title_select")
    if current_title is None:
        current_title = '1-2-Switch'

    # add weekly hours for most played game plot
    plot, content = game_of_the_week(source)
    plots.append((content, components(plot)))

    # add weekly hours distribution pie graph
    # TODO: if less than 3 games are played, RdBu palette fails
    plot, content = weekly_hours_snapshot(source, game_attr)
    plots.append((content, components(plot)))

    # add top 10 streaks graph
    plot, content = check_streaks(source)
    plots.append((content, components(plot)))

    # add weekly hours graph
    plot, content = line_weekly_hours(source)
    plots.append((content, components(plot)))

    # add rank by game time graph
    plot, content = bar_graph_top(source)
    plots.append((content, components(plot)))

    # add top game time pie graph
    plot, content = pie_graph_top(source)
    plots.append((content, components(plot)))

    # add top game line graph
    plot, content = line_graph_top(source)
    plots.append((content, components(plot)))

    # add weekly history (two games) graph
    plot, content = graph_two_games_weekly(source)
    plots.append((content, components(plot)))

    # single game view - hours
    plot, content = single_game(source, game_attr, current_title)
    dynamic.append((content, components(plot)))

    # single game view - streaks
    plot, content = single_game_streaks(source, current_title)
    # check if plot is null (no streaks)
    if plot == ['', '']:
        dynamic.append((content, plot))
    else:
        dynamic.append((content, components(plot)))

    # completion views
    # simple graph showing dates a game was completed
    plot, content = completion_dates(game_attr)
    complete.append((content, components(plot)))

    # display weekly game log
    week_log = weekly_log(source)

    # dataframe list of all games completed, reverse chronilogical
    games_completed = complete_list(game_attr)

    return render_template('gameplay.html', plots=plots,
                           game_log=game_log,
                           game_attr_demo=game_attr_demo,
                           source_sample=source_sample,
                           titles=titles,
                           current_title=current_title,
                           dynamic=dynamic,
                           complete=complete,
                           week_log=week_log,
                           games_completed=games_completed)


def init_game_data():
    '''
    Initializes data, returning two dataframes - source and complete
    '''
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    game_log = os.path.join(THIS_FOLDER, 'input/game_log.csv')
    game_attr = os.path.join(THIS_FOLDER, 'input/game_attr.csv')
    source = pd.read_csv(game_log,
                         parse_dates=['date', 'time_played'])
    game_attr = pd.read_csv(game_attr,
                           parse_dates=['date_completed',
                                        'release_date'])
    # storing initial csv states for how-to section
    game_log = source.head(3).to_html(index=False)
    game_attr_demo = game_attr.head(3).to_html(index=False)

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
    return source, game_attr, game_log, game_attr_demo


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
    weekly_top_games['date_title'] = (weekly_top_games['week_start']
                                      .dt.strftime('%Y-%m-%d') + ': '
                                      + weekly_top_games['title'])
    weekly_top_games = weekly_top_games.sort_values(['week_start'],
                                                    ascending=False)
    # add ranking field based on hours
    weekly_top_games['rank'] = (weekly_top_games['hours_played']
                                .rank(ascending=False))
    week_rank = int(weekly_top_games.head(1)['rank'].values)
    graph = weekly_top_games[['date_title',
                              'hours_played']].head(num_weeks)
    # plot graph df with bokeh
    source = ColumnDataSource(graph)
    y_range = sorted(set(graph['date_title']))

    plot = figure(plot_height=300,
                  sizing_mode='scale_width',
                  y_range=y_range,
                  title='Weekly Top Games',
                  toolbar_location='above',
                  tools='box_zoom,reset')
    plot.hbar(y='date_title',
              source=source,
              right='hours_played',
              height=.5,
              line_color='#8e8d7d',
              fill_color='#8e8d7d')
    plot.toolbar_location = None
    plot.axis.major_label_text_color = '#8e8d7d'
    plot.axis.axis_line_color = '#8e8d7d'
    plot.axis.major_tick_line_color = '#8e8d7d'
    plot.axis.minor_tick_line_color = '#8e8d7d'
    plot.title.text_color = '#8e8d7d'
    most_recent_week = weekly_top_games['week_start'].dt.date.iloc[0]
    most_recent_week = most_recent_week.strftime('%m-%d-%Y')
    curr_top_game = weekly_top_games['title'].iloc[0]
    top_game = (f'Top game for week of {most_recent_week}: '
                f'<em>{curr_top_game}</em>. ')
    # add 'out of num total weeks'
    total_weeks = str(len(weekly_top_games))
    top_game = (top_game + f'This ranks <em>{str(week_rank)}</em>'
                f'  out of <em>{total_weeks}</em> weeks.')
    top_game = '<h3>Game of the Week</h3>' + top_game
    return plot, top_game


def weekly_hours_snapshot(source, complete):
    '''
    Displays pie chart of time played per game in curretn week
    '''
    content = need_to_play(source, complete)
    df = __weekly_hours_by_game(source)
    # filter to current week
    current_week = df['week_start'].max()
    df = (df[df['week_start'] == current_week]
          [['title', 'hours_played']])
    # some calculations to orientate the pie wedges
    df['angle'] = (df['hours_played']/
                   df['hours_played'].sum() * 2*pi)
    # condition here to change palette if less than 3 or greater than 11
    if len(df['hours_played']) >= 3 and len(df['hours_played']) <= 11:
        df['color'] = RdBu[len(df['hours_played'])]
    else:
        df['color'] = cividis(len(df['hours_played']))
    source = ColumnDataSource(df)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title='Current Week Hours Distribution',
               toolbar_location=None,
               # tools='pan, reset',
               x_range=(-.5, 1.5))
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='title',
            fill_color='color', source=source)
    p.axis.visible = False
    p.title.text_color = '#8e8d7d'
    # p.legend.text_color = '#8e8d7d'
    p.legend.label_text_color  = '#8e8d7d'
    p.grid.grid_line_color = None
    return p, content


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
    content = '\nCurrent streak(s): '
    if len(current_streak) == 0:
        content = content + '<em>None</em>'
    else:
        current_streak['days'] = current_streak['streak_num'] + 1
        current_streak = current_streak[['title', 'days']]
        content = content + current_streak.to_html(index=False)
    # take max streak_sum, grouped by title, store in new object
    max_streak = df.groupby('title', as_index=False)['streak_num'].max()
    max_streak['days'] = max_streak['streak_num'] + 1
    max_streak = max_streak[['title', 'days']].sort_values(['days'],
                                                           ascending=False)
    # graph data
    # adapt this for bokeh, return plot
    max_streak['rank'] = max_streak['days'].rank(method='dense',
                                                 ascending=False)
    max_streak['rank'] = (max_streak['rank']
                          .astype(int)
                          .apply(lambda x: '{0:0>2}'
                          .format(x)))
    max_streak['rank_title'] = (max_streak['rank'].astype(str) +
                                '. ' + max_streak['title'])
    graph = max_streak[['rank_title', 'days']].head(top_games)
    # plot graph df with bokeh
    source = ColumnDataSource(graph)
    y_range = sorted(set(graph['rank_title']), reverse=True)
    title = 'Top ' + str(top_games) + ' Streaks'

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               y_range=y_range,
               title=title)
    p.hbar(y='rank_title',
           source=source,
           right='days',
           height=.5,
           line_color='#8e8d7d',
           fill_color='#8e8d7d')
    p.toolbar_location = None
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    content = '<h3>Streaks</h3>' + content
    return p, content


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
    # add current weekly hours for content text
    curr_week = graph['week_start'].max()
    curr_hrs = (graph[graph['week_start'] == curr_week]
                ['hours_played'].values[0])
    curr_hrs = str("{0:.1f}".format(curr_hrs))
    graph = graph.set_index('week_start')
    # add this as bokeh graph, return plot
    title = 'Weekly Hours'
    num_lines = len(graph.columns)
    my_palette = ['#8e8d7d', '#acd1e0']

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title,
               x_axis_type='datetime',
               tools='box_zoom, reset',
               toolbar_location='above')
    p.multi_line(xs=[graph.index.values]*num_lines,
                 ys=[graph[name].values for name in graph],
                 line_color=my_palette,
                 line_width=2)
    p.toolbar.logo = None
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    # show min/max of weekly and daily hours, averages
    content = '<h3>Overall Hours</h3>'
    # curr_weekly_hrs = str("{0:.1f}".format(graph.index.max()['hours_played']))
    avg_weekly_hrs = str("{0:.1f}".format(graph['hours_played'].mean()))
    max_weekly_hrs = str("{0:.1f}".format(graph['hours_played'].max()))
    min_weekly_hrs = str("{0:.1f}".format(graph['hours_played'].min()))
    content = content + f'<li><em>{curr_hrs}</em> hours played this week.</li>'
    content = (content + f'<li>On average, <em>{avg_weekly_hrs}</em> hours are ' +
                        'played per week.</li>')
    content = (content + '<li>The highest week had <em>'
               f'{max_weekly_hrs}</em> hours.</li>')
    content = (content + f'<li>The lowest week had <em>'
               f'{min_weekly_hrs}</em> hours.</li>')
    return p, content

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
    # top game at x hours, followed by 2nd game, hrs, 3rd game, hrs
    game1 = graph.iloc[[0]]['title'].values[0]
    game2 = graph.iloc[[1]]['title'].values[0]
    game3 = graph.iloc[[2]]['title'].values[0]
    hours1 = str("{0:.1f}".format(graph.iloc[[0]]
                                  ['hours_played'].values[0]))
    hours2 = str("{0:.1f}".format(graph.iloc[[1]]
                                  ['hours_played'].values[0]))
    hours3 = str("{0:.1f}".format(graph.iloc[[2]]
                                  ['hours_played'].values[0]))
    graph['title'] = (graph['rank'].astype(str)
                      + '. ' + graph['title'])
    # graph = graph['rank_title', 'hours_played']
    n_groups = graph['title'].count()
    game_rank = graph['hours_played']
    tick_names = graph['title']

    # add bokeh instructions, return plot
    source = ColumnDataSource(graph)
    y_range = sorted(set(graph['title']), reverse=True)
    title = 'Total Time Ranked by Game Title'

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               y_range=y_range,
               title=title,
               toolbar_location='above',
               tools='box_zoom,reset')
    p.hbar(y='title',
           source=source,
           right='hours_played',
           height=.5,
           line_color='#8e8d7d',
           fill_color='#8e8d7d')
    p.toolbar_location = None
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    # TODO: replace this with meaningful content
    content = '<h3>Top 3 Hours</h3>'
    # calc titles, hrs for top 3
    content = (content +
               f'<li><em>1st</em> is <em>{game1}</em> with <em>{hours1}</em> hours.</li>'
               f'<li><em>2nd</em> is <em>{game2}</em> with <em>{hours2}</em> hours.</li>'
               f'<li><em>3rd</em> is <em>{game3}</em> with <em>{hours3}</em> hours.</li>')
    return p, content


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
    top = pd.DataFrame(source_df[source_df['rank'] <= rank_num])
    top['rank'] = top['rank'].astype(int).apply(lambda x: '{0:0>2}'.format(x))
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
    # TODO: see if you can create a slider for the num_games
    graph = __current_top(__agg_total_time_played(source), num_games)
    game_rank = graph['hours_played']
    tick_names = graph['title']
    title = 'Top ' + str(num_games) + ' Distro'
    # add graph using bokeh, return plot
    graph['angle'] = (graph['hours_played']/
                      graph['hours_played'].sum() * 2*pi)
    graph['percent'] = (graph['hours_played']/
                        graph['hours_played'].sum())
    game1 = graph.iloc[[0]]['title'].values[0]
    game2 = graph.iloc[[1]]['title'].values[0]
    game3 = graph.iloc[[2]]['title'].values[0]
    percent1 = str("{0:.0%}".format(graph.iloc[[0]]
                                  ['percent'].values[0]))
    percent2 = str("{0:.0%}".format(graph.iloc[[1]]
                                  ['percent'].values[0]))
    percent3 = str("{0:.0%}".format(graph.iloc[[2]]
                                  ['percent'].values[0]))
    graph['color'] = RdBu[len(graph['hours_played'])]
    source = ColumnDataSource(graph)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title,
               toolbar_location=None,
               x_range=(-.5, 1.5))
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='title',
            fill_color='color', source=source)
    p.axis.visible = False
    p.legend.label_text_color  = '#8e8d7d'
    p.grid.grid_line_color = None
    p.title.text_color = '#8e8d7d'

    # TODO: add slider widget
    output_file("slider.html")
    slider = Slider(start=0, end=10, value=1, step=1, title="Stuff")
    widgetbox(slider)

    content = '<h3>Top 3 Ratio</h3>'
    content = (content + 'Of the top 10 most-played games:'
               f'<li><em>{game1}</em> is <em>{percent1}</em></li>'
               f'<li><em>{game2}</em> is <em>{percent2}</em></li>'
               f'<li><em>{game3}</em> is <em>{percent3}</em></li>')
    return p, content


def line_graph_top(source, rank_num=5):
    '''
    This graph shows every game that cracked the top rank number specified

    Parameters
    ----------
    rank_num : int
        Defines the range of rank numbers achieved, from 1 to rank_num.
    '''
    rank = __agg_time_rank_by_day(source)
    top = (rank[rank['rank_by_day'] <= rank_num])
    top_list = top['title']
    graph = (__agg_time_rank_by_day(source)
             [__agg_time_rank_by_day(source)['title']
              .isin(top_list)]
             [['date', 'title', 'rank_by_day']])
    # add to bokeh, return plot
    graph = (graph.groupby(['date', 'title'])
             .sum()['rank_by_day'].unstack())
    # add this as bokeh graph, return plot
    title = 'All Games That Reached Top ' + str(rank_num)
    num_lines = len(graph.columns)
    my_palette = Category20[len(graph.columns)]
    # TODO: look into upgrading bokeh to 1.0+ for legend
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title,
               x_axis_type='datetime',
               toolbar_location='above',
               tools='box_zoom,reset'
               )
    p.multi_line(xs=[graph.index.values]*num_lines,
                 ys=[graph[name].values for name in graph],
                 line_color=my_palette,
                 line_width=2)
    p.toolbar.logo = None
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    # TODO: replace this with meaningful content
    content = '<p></p>'
    return p, content


def __agg_time_rank_by_day(source):
    '''
    Ranks total game hours by game, by day
        - Calculates cumulative time spent in each game by day
        - Ranks each game by cumulative time for each day

    Returns
    -------
    time_rank : DataFrame
        date : datetime64[ns]
            Date of data sample.  This does not indicate that the game was
            played, it represents cumulative totals for the game at this
            date.
        title : object (str)
            Title of game
        cum_total_minutes : float64
            Cumulative minutes played for a given game, running from start
            of data tracking
        rank_by_day : float64
            Title's time ranked for given day
    '''
    time_rank_by_day = pd.DataFrame(source[['title', 'date',
                                            'cum_total_minutes']])
    game_list = pd.Series(source['title'].unique())
    date_list = pd.Series(source['date'].unique())

    date_game = pd.DataFrame(list(product(date_list, game_list)),
                             columns=['date', 'title'])
    time_rank = pd.DataFrame(date_game.merge(time_rank_by_day, how='left'))
    time_rank['rank_by_day'] = (time_rank.groupby('date')
                                ['cum_total_minutes']
                                .rank(method='dense', ascending=False))
    time_rank.sort_values(['title', 'date'], inplace=True)

    time_rank['cum_total_minutes'] = (time_rank.groupby('title')
                                      ['cum_total_minutes'].ffill())

    time_rank = time_rank[time_rank['cum_total_minutes'].notnull()]

    time_rank['rank_by_day'] = (time_rank.groupby('date')
                                ['cum_total_minutes']
                                .rank(method='dense', ascending=False))

    time_rank = time_rank.sort_values(['date', 'rank_by_day'])
    return time_rank


def graph_two_games_weekly(source,
                           game_title_1='Octopath Traveler',
                           game_title_2='Monster Hunter Generations'):
    '''
    Graphs the amount of time spent in hours for two games by week

    Returns
    -------
    graph_data : DataFrame
        Summary of time spent in hours on two games.  Fields below.

        index : datetime64[ns]
            Date game was played
        game_title_1 : float64
            Hours spent playing game for specified date
        game_title_2 : float64
            Hours spent playing game for specified date
    '''
    # TODO: modify this to accept a list, and remove single_game_weekly
    source_data = __weekly_hours_by_game(source)
    # pull data for game_title_1
    source_data_1 = source_data[(source_data['title'] == game_title_1)]
    graph_data_1 = (source_data_1[['week_start', 'hours_played']]
                    .set_index('week_start'))

    # pull data for game_title_2
    source_data_2 = source_data[(source_data['title'] == game_title_2)]
    graph_data_2 = (source_data_2[['week_start', 'hours_played']]
                    .set_index('week_start'))

    # combine data
    graph = graph_data_1.merge(graph_data_2, how='outer',
                                    left_index=True, right_index=True)

    # add in empty weeks
    start = graph.index.min()
    end = graph.index.max()
    d = pd.date_range(start, end)
    d = d[(d.weekday_name == 'Monday')]
    d = pd.DataFrame(d).set_index(0)
    graph = d.merge(graph, left_index=True, right_index=True,
                    how='left')
    graph.columns = [game_title_1, game_title_2]
    # add and return bokeh plot
    # TODO: add legend
    # TODO: add widget to choose games to compare
    title = 'Weekly History'
    num_lines = len(graph.columns)
    my_palette = ['#8e8d7d', '#acd1e0']

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=title,
               x_axis_type='datetime',
               toolbar_location='above',
               tools='box_zoom,reset')
    p.multi_line(xs=[graph.index.values]*num_lines,
                 ys=[graph[name].values for name in graph],
                 line_color=my_palette,
                 line_width=2)
    p.toolbar.logo = None
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    # TODO: replace this with meaningful content
    content = '<p></p>'
    return p, content


def need_to_play(source, complete, num_games=5):
    '''
    This returns a list of games that have not been completed and have
    not been played in the last 60 days.  Prints titles to console, with
    a heading "Consider playing:"

    Returns
    -------
    been_a_while : DataFrame
        Summary of games that haven't been played recently.  Fields below.

        title : object (str)
            Title of game
        date : datetime64[ns]
            Date last played
        dow : object (str)
            Day of week last played
        minutes_played : int64
            Cumulative minutes played for title
        hours_played : float64
            Cumulative hours played for title
        rank : float64
            Overall rank by time played
        complete : bool
            Indicates whether game has been completed (true) or not (false)
    '''
    last_played = last_played_game(source)
    hrs_played = __agg_total_time_played(source)
    # merge frames (inner join fine, both frames have the same 1-1 titles)
    been_a_while = last_played.merge(hrs_played, on='title')
    # narrow the list to games played longer than 2 months ago
    current_date = datetime.now()
    two_months_back = current_date - pd.Timedelta(days=60)
    been_a_while = been_a_while[been_a_while['date'] <= two_months_back]
    # check if game has been completed
    been_a_while = been_a_while.merge(complete, on='title')
    been_a_while = been_a_while[been_a_while['complete'] == False]
    been_a_while = been_a_while.sort_values(by='rank').head(num_games)
    been_a_while['days_since'] = datetime.now() - been_a_while['date']
    been_a_while['days_since'] = been_a_while['days_since'].astype('timedelta64[D]')
    # add hours, days since last played (iterate through frame)
    # adapt this to string, return
    playlist = '<h3>Consider Playing</h3>'
    for index, row in been_a_while.iterrows():
        playlist = playlist + ('<li><strong>' + row['title'] +
                    '</strong> (' +
                    str("{0:.0f}".format(row['days_since'])) +
                    ' days since, ' +
                    str("{0:.0f}".format(row['hours_played'])) +
                    ' hours in)</li>')
    return playlist


def last_played_game(source):
    '''
    Calculates the last played date of each game

    Returns
    -------
    last_played_by_game : DataFrame
        Summary of last played dates by game.  Fields below.

        title : object (str)
            Title of game
        date : datetime64[ns]
            Date game was last played
        dow : object (str)
            Day of week game was last played
    '''
    last_played_by_game = (source.groupby('title',
                                          as_index=False)
                           .max()[['title', 'date']])
    last_played_by_game = last_played_by_game.sort_values('date')
    last_played_by_game['dow'] = (last_played_by_game['date']
                                  .dt.weekday_name)
    return last_played_by_game


def single_game(source, complete, title):
    content = game_completed(complete, title)
    time, p = single_game_history(source, title)
    content = content + ' ' + time
    return p, content


def game_completed(completed, game_title):
    df = completed[completed['title'] == game_title]
    game_complete = df['complete'].values[0]
    if game_complete:
        # format time
        date_complete = pd.to_datetime(str(df['date_completed'].values[0]))
        date_complete = date_complete.strftime('%m-%d-%Y')
        # date_complete = date_complete.strftime('%Y-%m-%d')
        complete_status = (game_title + ' was <em>completed</em> on <em>'
                           + date_complete + '</em>.')
    else:
        complete_status = (game_title
                           + ' has <em>not</em> been <em>completed</em> yet.')
    complete_status = '<h3>Hours</h3>' + complete_status
    return complete_status


def create_hover_tool():
    """Generates the HTML for the Bokeh's hover data tool on the graph."""
    hover_html = """
      <div>
        <span class="hover-tooltip">$x</span>
      </div>
      <div>
        <span class="hover-tooltip">desc @description</span>
      </div>
    """
    return HoverTool(tooltips=hover_html)


def single_game_history(source, game_title):
    df = source[source['title'] == game_title]
    # add total hours spent playing game
    total_hours = df['hours_played'].sum()
    avg_playtime = str("{0:.1f}"
                       .format(df['hours_played'].mean()))
    playtime = f'Average playtime is <em>{avg_playtime}</em> hours.'
    playtime = playtime + ('  Played for a total of <em>'
                + str("{0:.1f}".format(total_hours))
                + '</em> hours ')
    # add hours for that week (if any)
    week_start = source['week_start'].max()
    weekly_hours = str("{0:.1f}"
                       .format(df[df['week_start'] == week_start]
                       ['hours_played'].sum()))
    if (weekly_hours != '0.0'):
        playtime = playtime + f' with <em>{weekly_hours}</em> hours played this week.'
    # create date range for graph
    # make range start from the 1st of the month on the min side
    min_date = df['date'].min().strftime('%Y-%m-01')
    date_range = pd.DataFrame(pd.date_range(min_date, df['date'].max()))
    date_range.columns = ['date']
    # format date shorter for graph (ax object?)
    # plt.locator_params(axis='x', nbins=10)
    df = pd.merge(df, date_range, how='right',
                  on='date').sort_values('date').reset_index()
    # get positions of start of each month, name/year of month
    # locs = df[df['date'].dt.day == 1].index
    # remove time from datetime
    # labels = (df[df['date'].dt.day == 1]['date'].values)
    graph = pd.DataFrame(df[['date', 'hours_played', 'description']])
    # add bokeh plot and return
    source = ColumnDataSource(graph)
    # print(source.data)
    title = game_title + ' Hours/Day'
    # top = graph['hours_played']
    # x_range = list(set(graph['date_str']))
    # TODO: add decsription field as tooltip
    # TOOLTIPS = [("summary", "@description")]

    # tools = []
    # hover_tool = create_hover_tool()
    # if hover_tool:
    tools = ['box_zoom,reset']
    tooltips = [('desc', '@description')]
    p = figure(plot_height=200,
               sizing_mode='scale_width',
               x_axis_type='datetime',
               title=title,
               toolbar_location='above',
               tools=tools,
               tooltips=tooltips)

    # hover = p.select_one(HoverTool)
    # hover.point_policy = "follow_mouse"

    p.vbar(x='date',
           source=source,
           width=2,
           top='hours_played',
           line_color='#8e8d7d',
           fill_color='#8e8d7d')

    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    p.toolbar.logo = None
    # show(p)
    return playtime, p


def single_game_streaks(source, game_title):
    '''
    Gives detailed information on gameplay streaks for specified game_title
    '''
    df = get_streaks(source)
    # import pdb; pdb.set_trace()
    df = df[df['title'] == game_title]
    # check for 0 streaks to avoid errors
    if len(df) != 0:
        # init this to true for first loop
        first_streak = True
        streak_ranges = pd.DataFrame(columns=['start', 'end'])
        # loop through streak_num col, starting with 1 until next 1 reached
        # for index, row in df.iterrows():
        for i, (index, row) in enumerate(df.iterrows()):
            # import pdb; pdb.set_trace()
            # record the date at 1, and also last next_day before next 1
            # two things would trigger logging next_day:
            # 1. we hit streak_num = 1 after first streak_num = 1
            # 2. we hit the end of the dataframe
            start = row['date']
            streak = row['streak_num']
            if streak == 1.0:
                # need to find out if this is the first streak for logic
                if first_streak is False:
                    last_df = end
                    # append date at 1 and last next_day to dataframe
                    add_row = pd.DataFrame([(start_df, last_df)],
                                           columns=['start', 'end'])
                    streak_ranges = pd.concat([streak_ranges, add_row])
                # no matter what, start begins here
                start_df = start
                # first_streak = False
            # repeat until end of dataframe is reached
            # check for end of df
            if i == len(df) - 1:
                last_df = row['next_day']
                add_row = pd.DataFrame([(start_df, last_df)],
                                       columns=['start', 'end'])
                streak_ranges = pd.concat([streak_ranges, add_row])
            first_streak = False
            # because of the algorithm's lag, this needs to be logged last
            end = row['next_day']
        # create column for number of days for each streak
        streak_ranges['days'] = (streak_ranges['end']
                                 - streak_ranges['start'])
        # create column for rank based on days for each streak
        streak_ranges['rank'] = streak_ranges['days'].rank(ascending=False,
                                                           method='dense')
        max_days = (streak_ranges[streak_ranges['rank'] == 1][['days']]
                    .values)
        max_start = (streak_ranges[streak_ranges['rank'] == 1][['start']]
                     .values)
        max_end = streak_ranges[streak_ranges['rank'] == 1][['end']].values
        streak_output = '<em>' + str(len(streak_ranges)) + '</em> streak(s).  '
        # fix print out summary of streaks - maximum, total num, etc
        max_days = int(max_days[0][0] / np.timedelta64(1, 'D')) + 1
        max_start = (pd.to_datetime(str(max_start[0][0]))
                     .strftime('%m-%d-%Y'))
        max_end = (pd.to_datetime(str(max_end[0][0]))
                     .strftime('%m-%d-%Y'))
        # TODO: modify to display current streaks if they are longest
        streak_output = (streak_output + 'The longest streak '
                         f'played was for <em>{max_days}</em> days, ' +
                         f'starting on <em>{max_start}</em> and ' +
                         f'running until <em>{max_end}</em>.')

        # create graph of streaks and display
        streak_dates = pd.Series()
        for i, (index, row) in enumerate(streak_ranges.iterrows()):
            # start by create date series between each start and end
            start = row['start']
            end = row['end']
            new_range = pd.date_range(start, end)
            streak_dates = streak_dates.append(new_range.to_series())

        graph_data = pd.DataFrame(streak_dates)
        # create value for graphing, and remove extra date column
        graph_data['played'] = 1
        graph_data = graph_data['played']
        # create dates for gaps between streaks
        all_days = (pd.DataFrame(pd.date_range(start=streak_dates.min(),
                                               end=streak_dates.max()))
                    .set_index(0))
        # join on index with graph_data
        graph = all_days.join(graph_data, how='left').reset_index()
        graph.columns = ['date', 'played']
        # graph_data_final.plot(title='Gameplay Streaks')
        # add bokeh graph, return plot, content
        title = game_title + ' Gameplay Streaks'
        source = ColumnDataSource(graph)
        # num_lines = len(graph.columns)

        p = figure(plot_height=200,
                   sizing_mode='scale_width',
                   title=title,
                   x_axis_type='datetime',
                   toolbar_location='above',
                   tools='box_zoom,reset'
                   )
        p.line(source=source,
               x='date',
               y='played',
               line_color='#8e8d7d',
               line_width=2)
        p.yaxis.visible = False
        p.axis.major_label_text_color = '#8e8d7d'
        p.axis.axis_line_color = '#8e8d7d'
        p.axis.major_tick_line_color = '#8e8d7d'
        p.axis.minor_tick_line_color = '#8e8d7d'
        p.title.text_color = '#8e8d7d'
        p.toolbar.logo = None
    else:
        streak_output = 'No streaks.'
        p = ['','']
    streak_output = '<h3>Streaks</h3>' + streak_output
    return p, streak_output


def completion_dates(df):
    # calculate 5 most recently completed games
    df = df.sort_values('date_completed', ascending=False)
    recent = df.head(5)[['title', 'date_completed']]
    content = 'Recently completed:<br>' + recent.to_html(index=False)
    df = df[df['date_completed'].notnull()]
    # focus on the last 10 years
    first_year = df['date_completed'].max() - pd.DateOffset(years=10)
    df = df[df['date_completed'].dt.year >= first_year.year]
    df['value'] = 1
    graph = pd.DataFrame(df[['date_completed', 'value']])
    # add bokeh plot and return
    source = ColumnDataSource(graph)
    title = 'Completion Dates (Last 10 Years)'

    p = figure(plot_height=300,
               sizing_mode='scale_width',
               x_axis_type='datetime',
               title=title,
               toolbar_location='above',
               tools='box_zoom,reset')
    p.vbar(x='date_completed',
           source=source,
           width=2,
           top='value',
           line_color='#8e8d7d',
           fill_color='#8e8d7d')
    p.axis.major_label_text_color = '#8e8d7d'
    p.axis.axis_line_color = '#8e8d7d'
    p.axis.major_tick_line_color = '#8e8d7d'
    p.axis.minor_tick_line_color = '#8e8d7d'
    p.title.text_color = '#8e8d7d'
    p.toolbar.logo = None
    return p, content


def complete_list(df):
    df = df.sort_values('date_completed', ascending=False)
    df = df[df['date_completed'].notnull()][['title', 'date_completed']]
    return df.to_html(index=False)


def weekly_log(df):
    current_week = df['week_start'].max()
    df = df[df['week_start'] == current_week].sort_values('date')
    df = df[['date', 'title', 'hours_played', 'description']]
    df['hours_played'] = df['hours_played'].round(2)
    content = df.to_html(index=False)
    return content

@app.route('/music/')
def music():
    # init dashboard element list, data
    plots = []
    df, df_apple, file_path = init_music_data()

    # test for first graph
    plot, content = graph_never_listened(10, df_apple)
    plots.append((content, components(plot)))
    return render_template('music.html', plots=plots)


def init_music_data():
    '''
    Initializes data frames, importing itunes' playlist csv export, and then
    returning data for processing.

    Returns
    -------
    df_all : data frame
        Entire contents of csv file
    df_apple_music : data frame
        Subset of df_all, containing only songs in Apple's AAC format
    data_path : string
        The folder path to the itunes csv file
    '''
    # read in and format data
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(THIS_FOLDER, 'input/music.csv')
    df_all = pd.read_csv(data_path,
                         parse_dates=['Date Modified', 'Date Added',
                                      'Last Played', 'Last Skipped'])
    # dfRaw = pd.read_csv(data_path + 'test data/Music.csv')
    focus_fields = ['Artist', 'Name', 'Album', 'Genre', 'Plays',
                    'Date Added', 'Last Played', 'Year']
    # dateFields = ['Date Modified','Date Added','Last Played','Last Skipped']
    # df_all.sort_values(by='Date Added', ascending=False, inplace=True)
    # this fixes the formats that auto parse failed on
    # might be a sign there's an issue importing
    df_all['Date Modified'] = pd.to_datetime(df_all['Date Modified'],
                                             format='%m/%d/%y, %I:%M %p',
                                             errors='coerce')
    df_all['Date Added'] = pd.to_datetime(df_all['Date Added'],
                                          format='%m/%d/%y, %I:%M %p',
                                          errors='coerce')
    df_all['Year'] = pd.to_numeric(df_all['Year'], errors='coerce')

    # create some subsets of data
    # look at genres of music
    # may want to filter out manual tags by file format
    df_apple_music = pd.DataFrame(df_all[(df_all['Kind'] ==
                                          'Apple Music AAC audio file')])
    df_apple_music.sort_values(by=['Plays'], inplace=True, ascending=False)
    df_apple_music = df_apple_music[focus_fields]

    df_apple_music_genre_play_counts = (df_apple_music.groupby(['Genre'])
                                        [['Genre', 'Plays']].sum())
    df_apple_music_genre_play_counts.sort_values(by=['Plays'],
                                                 ascending=False, inplace=True)

    return df_all, df_apple_music, data_path


def graph_never_listened(num, df_apple):
    '''
    Creates a pie graph  of artists with songs that haven't been listened to.
    Larger slices represent more songs.
    '''
    # TODO: add function
    df_apple_music_song_play_counts = song_play_counts(df_apple)
    # TODO: add function
    songs_never_listened = never_listened(df_apple_music_song_play_counts)
    song_counts = (songs_never_listened[['Artist', 'Name']]
                   .groupby('Artist', as_index=False)
                   .count())
    song_counts['Rank'] = song_counts['Name'].rank(ascending=False,
                                                   method='min')
    song_counts.sort_values('Rank', inplace=True)
    song_counts = song_counts.head(num)
    # dfTop = currentTop(numGames)
    # add bokeh graph
    # some calculations to orientate the pie wedges
    song_counts['angle'] = (song_counts['Name']/
                            song_counts['Name'].sum() * 2*pi)
    song_counts['color'] = RdBu[len(song_counts['Name'])]
    source = ColumnDataSource(song_counts)
    p = figure(plot_height=300,
               sizing_mode='scale_width',
               title=f'Need to Listen - Top {num}',
               toolbar_location=None,
               # tools='pan, reset',
               x_range=(-.5, 1.5))
    p.wedge(x=0, y=1, radius=0.4, line_color='white',
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'), legend='Artist',
            fill_color='color', source=source)
    p.axis.visible = False
    p.title.text_color = '#8e8d7d'
    # p.legend.text_color = '#8e8d7d'
    p.legend.label_text_color  = '#8e8d7d'
    p.grid.grid_line_color = None
    content = '<p>Test Graph</p>'
    return p, content


def song_play_counts(source_df):
    '''
    Gives artist, song name, and total play counts
    '''
    return_df = pd.DataFrame(source_df
                             .groupby(['Artist', 'Name'])
                             [['Artist', 'Name', 'Plays']]
                             .sum().reset_index())
    return_df.sort_values(by=['Plays'], ascending=False,
                          inplace=True)
    return_df.name = 'df_apple_music_song_play_counts'
    return return_df


def never_listened(source_df):
    '''
    Gives the artist and song name of songs that have nevery been played
    '''
    return_df = (pd.DataFrame
                 (source_df
                  [(source_df['Plays'] == 0)]))
    return_df.sort_values(by='Artist', inplace=True)
    return_df.name = 'songs_never_listened'
    return return_df


@app.route('/exercise/')
def exercise():
    return render_template('exercise.html')
