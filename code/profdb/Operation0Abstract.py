import os, logging, duckdb
from abc import ABC, abstractmethod

class OperationAbstract(ABC):
    def __init__(self, db_path: str):
        self.con_path = db_path
        self.con = duckdb.connect(db_path)
        self.sample = False

    def pre_abstract(self):
        """Always run before pre()."""
        logging.info(f"Operation {self.__class__.__name__} starting.")

    def pre(self):
        """Pre checks and setup to be implemented in subclasses."""
        ...

    def trans(self):
        """Main transformation step to be implemented in subclasses."""
        ...

    def post_abstract(self):
        logging.info(f"Operation {self.__class__.__name__} completed.")

    def run(self):
        try:
            self.pre_abstract()
            self.con.execute("BEGIN")
            self.pre()
            self.trans()
            self.con.execute("COMMIT")
            self.post_abstract()

        except Exception as e:
            # 1) Try rollback, but never let rollback errors mask the real error
            try:
                self.con.execute("ROLLBACK")
            except Exception:
                logging.exception("ROLLBACK failed (ignoring to show original error)")

            # 2) Print full stack trace to logs
            logging.exception("Pipeline failed with an exception")  # includes stack trace

            # 3) Keep your existing failure handler
            self._quit_on_failure(f"Pipeline failed: {e}")

            # 4) Re-raise original exception with its traceback
            raise

        finally:
            try:
                if getattr(self, "con", None):
                    self.con.close()
            except Exception:
                logging.exception("Failed to close DB connection")

    def _quit_on_failure(self, error_msg=None):
        if error_msg:
            logging.error(error_msg)
        if self.con:
            self.con.close()
        logging.error("Quitting due to failure.")
        raise SystemExit(1)
