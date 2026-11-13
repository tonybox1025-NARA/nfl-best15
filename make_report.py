"""output/latest.json → 읽기 쉬운 마크다운 리포트."""
import json
from pathlib import Path

r = json.loads(Path("output/latest.json").read_text())
L = [f"# {r['season']} 시즌 {r['week']}주차 베스트 15", "",
     f"생성: {r['generated_at']}  ", "데이터: " + " / ".join(r["sources"].values()), "",
     "> 확률은 과거 데이터 기반 추정치예요. [ ]는 부상·훈련 상태예요. 출전 여부가 확정되면 다시 업데이트하세요.", ""]

def table(rows, td=False):
    out = ["| 순위 | 선수 | 팀 | 상대 | 픽 | 확률 | 상태 |", "|---|---|---|---|---|---|---|"]
    for p in rows:
        out.append(f"| {p['rank']} | {p['name']} ({p['pos']}) | {p['team']} | {p['opp']} | {p['label']} | **{p['prob']}%** | {p['status']} |")
    return out

L += ["## 전체 경기 베스트 15 (확률순)", ""] + table(r["overall"]["prob"])
L += ["", "## 터치다운 확률 TOP 15", ""] + table(r["overall"]["td"])
L += ["", "## 경기별 베스트 15", ""]
for g in r["games"]:
    L += [f"### {g['away']} @ {g['home']} · {g['kickoff']} · {g['stadium']}",
          f"스프레드 {g['home']} {-g['spread_home']:+} · 총점 {g['total']} · 예상 점수 {g['away']} {g['proj']['away']} – {g['home']} {g['proj']['home']}", ""]
    L += table(g["best15_prob"]) + [""]
Path(f"output/week{r['week']}_report.md").write_text("\n".join(L), encoding="utf-8")
print(f"output/week{r['week']}_report.md")
