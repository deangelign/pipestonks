import investpy  # type: ignore
import yfinance as yf  # type: ignore
from typing import List, Tuple, Union
from datetime import datetime, timedelta


def list_all_stock_symbols() -> List[str]:
    """Returns a list of all stock symbols.

    Returns:
        A list of all stock symbols.
    """
    list_symbols = investpy.stocks.get_stocks_list(country="Brazil")
    list_symbols.sort()
    list_symbols = [x + ".SA" for x in list_symbols]
    return list_symbols


def get_range_dates(left_offset: int = 0, right_offset: int = 0) -> Tuple[str, str]:
    """Returns a tuple of datetime objects representing a range of dates.

    with optional offsets from the current date.
    Args:
        left_offset (int, optional): The number of days to subtract from the current date to get
            the starting date of the range. Defaults to 0.

        right_offset (int, optional): The number of days to add to the current date to get the
            ending date of the range. Defaults to 0.

    Returns:
        Tuple[str, str]: A tuple of two strings representing the start and end
            of the date range.
    """
    left = (datetime.now() + timedelta(days=left_offset)).strftime("%Y-%m-%d")
    right = (datetime.now() + timedelta(days=right_offset)).strftime("%Y-%m-%d")

    start_time = str(left)
    end_time = str(right)
    return (start_time, end_time)


def retrieve_stocks_data(
    stock_name: Union[str, List[str]],
    period: str = None,
    date_range: Tuple[str, str] = None,
):
    """Retrieves stock data for a given stock symbol or a list of symbols.

    Args:
        stock_name (Union[str, List[str]]): A stock symbol or a list of symbols for which data is
            to be retrieved.

        period (str, optional): The time period to retrieve data for. The available options
            are: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. Defaults to None.

        date_range (Tuple[datetime, datetime], optional): A tuple of two strings
            specifying the start and end dates of the date range to retrieve data for.
            Defaults to None.

    Returns:
        pandas.DataFrame: A Dataframe containing the retrieved stock data.

    Raises:
        ValueError: If neither `period` nor `date_range` is set.
    """
    if period:
        data = yf.download(
            tickers=stock_name,
            period=period,
            interval="1d",
            ignore_tz=False,
            # group_by = 'ticker',
            auto_adjust=True,
            repair=False,
            prepost=True,
            threads=True,
            proxy=None,
        )
    elif date_range:
        data = yf.download(
            tickers=stock_name,
            start=date_range[0],
            end=date_range[1],
            interval="1d",
            ignore_tz=False,
            # group_by = 'ticker',
            auto_adjust=True,
            repair=False,
            prepost=True,
            threads=True,
            proxy=None,
        )
    else:
        raise ValueError(
            "Neither `period` nor `date_range` is set. Please set a value for one of them."
        )

    return data
