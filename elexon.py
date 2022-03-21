from collections import defaultdict
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


class ElexonReport(pydantic.BaseModel):
    report: str
    columns: list


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


if __name__ == "__main__":
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

    dataset = defaultdict(list)
    for rep in reports:
        for date in pd.date_range("2020-01-01", "2020-01-03", freq="D"):
            dataset[rep.report].append(
                send_elexon_request(ElexonRequest(report=rep.report, date=date))
            )

    for report, data in dataset.items():
        data = pd.concat(data, axis=0)
        data.to_csv(f"./data/{report}-all.csv")
        print(f"combined {len(data)} days for {report} into {data.shape}")

    dataset = {}
    for rep in reports:
        data = pd.read_csv(f"./data/{rep.report}-all.csv").loc[:, rep.columns]

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
        dataset[rep.report] = data

    final = pd.concat(dataset.values(), axis=1)
    final = final.loc[:, ~final.columns.duplicated()]
    final.to_csv("./data/final.csv")
