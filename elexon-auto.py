import datetime
import pydantic
from secret import api_key


class ElexonRequest(pydantic.BaseModel):
    report: str
    date: datetime.date
    api_key: pydantic.SecretStr = pydantic.SecretStr(api_key)
    service_type: str = "csv"


req = ElexonRequest(report="B1770", date="2020-01-01")
"""
report='B1770' api_key=SecretStr('**********') date=datetime.date(2020, 1, 1) service_type='csv'
"""
print(req.report, req.date, req.service_type)
"""
B1770 2020-01-01 csv
"""
url = f"https://api.bmreports.com/BMRS/{req.report}/v1?APIKey={req.api_key}&Period=*&SettlementDate={req.date.isoformat()}&ServiceType={req.service_type}"
"""
https://api.bmreports.com/BMRS/B1770/v1?APIKey=**********&Period=*&SettlementDate=2020-01-01&ServiceType=csv
"""
import requests

req = ElexonRequest(report="B1770", api_key=api_key, date="2020-01-01")
url = f"https://api.bmreports.com/BMRS/{req.report}/v1?APIKey={req.api_key.get_secret_value()}&Period=*&SettlementDate={req.date.isoformat()}&ServiceType={req.service_type}"
res = requests.get(url)
assert res.status_code == 200
print(res.text[:512])
"""
*
*
*Imbalance Prices Service For Balancing (B1770) Data
*
*DocumentID,DocumentRevNum,ActiveFlag,ProcessType,DocumentType,Resolution,CurveType,PriceCategory,ImbalancePriceAmount,SettlementPeriod,SettlementDate,ControlArea,BusinessType,TimeSeriesID,DocumentStatus
ELX-EMFIP-IMBP-22444358,1,Y,Realised,Imbalance prices,PT30M,Sequential fixed size block,Excess balance,49.95,48,2020-01-01,10YGB----------A,Balance energy deviation,ELX-EMFIP-IMBP-TS-2,Final
ELX-EMFIP-IMBP-22444358,1,Y,Realised,Imbalance prices,PT30
"""
from pathlib import Path

fi = Path().cwd() / "data" / f"{req.report}-{req.date}.csv"
fi.parent.mkdir(exist_ok=True)
fi.write_text(res.text)
import pandas as pd

data = pd.read_csv("./data/B1770-2020-01-01.csv", skiprows=4)
print(data.iloc[:3, :7])
"""
               *DocumentID  DocumentRevNum ActiveFlag ProcessType
0  ELX-EMFIP-IMBP-22444358             1.0          Y    Realised
1  ELX-EMFIP-IMBP-22444358             1.0          Y    Realised
2  ELX-EMFIP-IMBP-22444205             1.0          Y    Realised
"""
print(data.columns)
"""
Index(['*DocumentID', 'DocumentRevNum', 'ActiveFlag', 'ProcessType',
       'DocumentType', 'Resolution', 'CurveType', 'PriceCategory',
       'ImbalancePriceAmount', 'SettlementPeriod', 'SettlementDate',
       'ControlArea', 'BusinessType', 'TimeSeriesID', 'DocumentStatus'],
      dtype='object')
"""
print(data.iloc[-3:, :4])
"""
                *DocumentID  DocumentRevNum ActiveFlag ProcessType
94  ELX-EMFIP-IMBP-22437930             1.0          Y    Realised
95  ELX-EMFIP-IMBP-22437930             1.0          Y    Realised
96                    <EOF>             NaN        NaN         NaN
"""
data = data.dropna(axis=0, subset=["SettlementDate"])
print(data.iloc[-3:, :4])
"""
                *DocumentID  DocumentRevNum ActiveFlag ProcessType
93  ELX-EMFIP-IMBP-22438072             1.0          Y    Realised
94  ELX-EMFIP-IMBP-22437930             1.0          Y    Realised
95  ELX-EMFIP-IMBP-22437930             1.0          Y    Realised
"""
import io

buff = io.StringIO(res.content.decode("UTF-8"))
data = pd.read_csv(buff, skiprows=4)
import datetime
from pathlib import Path

import requests
import pandas as pd
import pydantic

from secret import api_key


class ElexonRequest(pydantic.BaseModel):
    report: str
    date: datetime.date
    api_key: pydantic.SecretStr = pydantic.SecretStr(api_key)
    service_type: str = "csv"


def send_elexon_request(req: ElexonRequest) -> pd.DataFrame:
    url = f"https://api.bmreports.com/BMRS/{req.report}/v1?APIKey={req.api_key.get_secret_value()}&Period=*&SettlementDate={req.date.isoformat()}&ServiceType={req.service_type}"
    res = requests.get(url)
    assert res.status_code == 200

    fi = Path().cwd() / "data" / f"{req.report}-{req.date}.csv"
    fi.parent.mkdir(exist_ok=True)
    fi.write_text(res.text)

    data = pd.read_csv(fi, skiprows=4)
    #  B1770 has SettlementDate, B1780 has Settlement Date
    data.columns = [d.replace(" ", "") for d in data.columns]
    return data.dropna(axis=0, subset=["SettlementDate"])


data = send_elexon_request(ElexonRequest(report="B1770", date="2020-01-01"))
from collections import defaultdict

dataset = defaultdict(list)
for report in ["B1770", "B1780"]:
    for date in pd.date_range("2020-01-01", "2020-01-03", freq="D"):
        data = send_elexon_request(ElexonRequest(report=report, date=date))
        dataset[report].append(data)
for report, data in dataset.items():
    data = pd.concat(data, axis=0)
    data.to_csv(f"./data/{report}-all.csv")
    print(f"combined {len(data)} days for {report} into {data.shape}")


class ElexonReport(pydantic.BaseModel):
    report: str
    columns: list


rep = ElexonReport(
    report="B1770",
    columns=[
        "SettlementDate",
        "SettlementPeriod",
        "ImbalancePriceAmount",
        "PriceCategory",
    ],
)

data = pd.read_csv(f"./data/{rep.report}-all.csv")

data = (
    data.pivot(
        index=["SettlementDate", "SettlementPeriod"],
        columns="PriceCategory",
        values="ImbalancePriceAmount",
    )
    .sort_index()
    .reset_index()
)
data["datetime"] = pd.date_range(
    start=data["SettlementDate"].min(),
    periods=len(data),
    freq="30T",
    tz="Europe/London",
)
reports = [
    ElexonReport(
        report="B1770",
        columns=[
            "SettlementDate",
            "SettlementPeriod",
            "ImbalancePriceAmount",
            "PriceCategory",
        ],
    ),
    ElexonReport(
        report="B1780",
        columns=[
            "SettlementDate",
            "SettlementPeriod",
            "ImbalanceQuantity(MAW)",
            "ImbalanceQuantityDirection",
        ],
    ),
]


dataset = {}
for rep in reports:
    data = pd.read_csv(f"./data/{rep.report}-all.csv")
    data = data.loc[:, rep.columns]

    if rep.report == "B1770":
        data = (
            data.pivot(
                index=["SettlementDate", "SettlementPeriod"],
                columns="PriceCategory",
                values="ImbalancePriceAmount",
            )
            .sort_index()
            .reset_index()
        )

    data = data.sort_values(["SettlementDate", "SettlementPeriod"])

    data["datetime"] = pd.date_range(
        start=data["SettlementDate"].min(),
        periods=len(data),
        freq="30T",
        tz="Europe/London",
    )
    data = data.set_index("datetime")
    print(data.head(2))
    dataset[rep.report] = data

final = pd.concat(dataset.values(), axis=1)
print(final.head(3))
final = final.loc[:, ~final.columns.duplicated()]
print(final.head(3))

"""
SettlementDate  SettlementPeriod  Excess balance  Insufficient balance  ImbalanceQuantity(MAW) ImbalanceQuantityDirection
datetime
2020-01-01 00:00:00+00:00     2020-01-01               1.0        50.90000              50.90000                 54.3365                    SURPLUS
2020-01-01 00:30:00+00:00     2020-01-01               2.0        51.00000              51.00000                194.7133                    SURPLUS
2020-01-01 01:00:00+00:00     2020-01-01               3.0        29.37006              29.37006                -71.4292                    DEFICIT
"""
