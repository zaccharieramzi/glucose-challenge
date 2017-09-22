from io import StringIO
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model

motion_path = "motion.tsv"
df_motion = pd.read_csv(motion_path, sep="\t", header=None, index_col=0)
df_motion.index = pd.to_datetime(df_motion.index, infer_datetime_format=True)
analysis_time_span = pd.to_timedelta(48, unit="h")

walking_time = np.genfromtxt("walking_time.csv")
glucose_stddev = np.genfromtxt("glucose_stddev.csv")
X_train = np.expand_dims(walking_time, 1)
Y_train = (glucose_stddev > 12.5).astype(int)

# In this case we use logistic regression since we want to output a risk score.
regressor = linear_model.LogisticRegression(C=1)
regressor.fit(X_train, Y_train)


# Lot of copy-paste, this could go away by refactoring
def time_in_activity_between(s, start, stop):
    """Compute the time spent in a certain activity between start and stop time.

    This function aggregates the time spent in a certain activity given by a pd
    Series between a start and stop time. It will consider the activity
    happening before the time span to see if we begin by walking.

    Arguments:
        - s (pd.Series): the series containing the activity state with the time
        in index.
        - start (pd.TimeStamp): the beginning of the period of interest.
        - stop (pd.TimeStamp): the end of the period of interest.

    Returns:
        - float: the time spent in activity expressed in minutes.
    """
    res = 0
    time_prev = 0
    state_prev = 0
    one_minute = pd.offsets.Minute(1)
    s_interest = s[start:stop]
    if len(s_interest) > 0:
        # Let's look at what was happening before
        first_time_interest = s_interest.index[0]
        iloc_time = s.index.get_loc(first_time_interest)
        if iloc_time > 0:
            if s[iloc_time - 1] == 1:
                time_prev = start
                state_prev = 1
        # Now let's iterate through the rows
        for time, state in s_interest.iteritems():
            if state_prev == 1 and state == 0:
                res += (time - time_prev) / one_minute
            if state == 0:
                time_prev = time
                state_prev = 0
            if state == 1 and state_prev == 0:
                time_prev = time
                state_prev = 1
        # Let's account for the last activity
        if state_prev == 1:
            res += (stop - time_prev) / one_minute
    else:
        # nothing is changing in that time span
        last_state_index = s[s.index < start].index[-1]
        last_state = s[last_state_index]
        if last_state == 1:
            res = (stop - start) / one_minute
    return res


for line in sys.stdin:
    new_data = StringIO(line)
    new_df_motion = pd.read_csv(new_data, sep="\t", header=None, index_col=0)
    new_df_motion.index = pd.to_datetime(
        new_df_motion.index, infer_datetime_format=True)
    df_motion = pd.concat([df_motion, new_df_motion])
    df_walking = df_motion[2]
    last_date = df_walking.index[-1]
    new_X = time_in_activity_between(
            df_walking, last_date - analysis_time_span, last_date)
    res = regressor.predict_proba([[new_X]])
    print(res[0, 1] * 100)
