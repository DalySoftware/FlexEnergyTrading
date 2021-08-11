import sys
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from NaturalGasDataProvider import NaturalGasDataProvider
from pandas import DataFrame  # type: ignore


class GruAgent:
    @staticmethod
    def scale_minus_one_to_one(data_frame: DataFrame, column: str):
        result = data_frame.copy()

        min_max_scaler = MinMaxScaler()

        result[column] = min_max_scaler.fit_transform(
            result[column].values.reshape(-1, 1)
        )

        return result

    @classmethod
    def normalize_data(cls, data_frame: DataFrame) -> DataFrame:
        result = data_frame.copy()

        result = cls.scale_minus_one_to_one(result, "Open")
        result = cls.scale_minus_one_to_one(result, "Close")
        result = cls.scale_minus_one_to_one(result, "High")
        result = cls.scale_minus_one_to_one(result, "Low")
        result = cls.scale_minus_one_to_one(result, "Volume")

        return result


if __name__ == '__main__':
    print(sys.path)
    print(sys.version)

    data = NaturalGasDataProvider.get_data()

    print(data.tail())

    scaled = GruAgent.normalize_data(data)

    print(scaled.tail())
