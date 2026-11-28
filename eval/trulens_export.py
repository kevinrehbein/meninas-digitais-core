from trulens.core import TruSession

session = TruSession()
print(session.get_leaderboard())                 # o mesmo que o painel
records, feedback_cols = session.get_records_and_feedback()
print(records[feedback_cols].mean())