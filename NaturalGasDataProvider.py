from numpy.lib.function_base import append
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_ta as ta  # type: ignore
from datetime import datetime


class NaturalGasDataProvider:
    @staticmethod
    def _convert_volume(vol_string: str):
        if vol_string == "-":
            return 0

        return 1000 * float(vol_string.replace("K", ""))

    @classmethod
    def get_data(cls, add_ta_lib=False) -> DataFrame:
        df = pd.read_csv(
            ".\\InputData\\Natural Gas Futures Historical Data 1997 - 2021.csv")
        df["Date"] = df["Date"].apply(
            lambda x: datetime.strptime(x, "%b %d, %Y"))
        df["Vol."] = df["Vol."].apply(cls._convert_volume)

        df = df.drop(columns=["Change %"])
        df = df.rename(columns={"Price": "Close"})
        df = df.rename(columns={"Vol.": "Volume"})

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        df.set_index("Date", inplace=True)  # type: ignore
        df.sort_index(inplace=True)

        # df.ta.log_return(cumulative=True, append=True)
        # df.ta.percent_return(cumulative=True, append=True)
        # df.ta.sma(length=10, append=True)

        # df["GC"] = df.ta.sma(50, append=True) > df.ta.sma(200, append=True)
        # df.ta.tsignals(df.GC, asBool=True, append=True)

        if (add_ta_lib):
            df.ta.strategy(ta.AllStrategy, append=True)  # type: ignore

        return df


def main():
    data = NaturalGasDataProvider.get_data()

    print(data.tail())
    print(data.shape)


if __name__ == '__main__':
    main()
