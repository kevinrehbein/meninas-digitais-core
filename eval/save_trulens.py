from trulens.core import TruSession

session = TruSession()
records, feedback_cols = session.get_records_and_feedback()
records.to_csv("eval/trulens_resultados.csv", index=False, encoding="utf-8-sig")