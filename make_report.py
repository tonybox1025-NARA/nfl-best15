"""output/latest.json → 읽기 쉬운 마크다운 리포트."""
import json
from pathlib import Path

r = json.loads(Path("output/latest.json").read_text())
L = [f"# {r['season']} 시즌 {r['week']}주차 베스트 15", "",
     f"생성: {r['generated_at']}  ", "데이터: " + " / ".join(r["sources"].values()), "",
     "> 확률은 과거 데이터 기반 추정치예요. 배당이 있는 픽은 모델 65% + 시장 35%로 섞은 확률이에요. ⚠는 모델과 시장 차이가 15%p 넘는 픽(역할 변화·부상 확인 필요)이에요.", ""]

def table(rows, td=False):
    out = ["| 순위 | 선수 | 팀 | 상대 | 픽 | 확률 | FanDuel | 엣지 | 상태 |", "|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        fd = f"{p['odds']:+.0f}" if p.get("odds") is not None else "-"
        ed = f"{p['edge']:+.1f}%p" if p.get("edge") is not None else "-"
        st = p["status"] + (" ⚠시장과 차이 큼" if p.get("check") else "")
        out.append(f"| {p['rank']} | {p['name']} ({p['pos']}) | {p['team']} | {p['opp']} | {p['label']} | **{p['prob']}%** | {fd} | {ed} | {st} |")
    return out

L += ["## 전체 경기 베스트 15 (확률순)", ""] + table(r["overall"]["prob"])
if r["overall"].get("edge"):
    L += ["", "## 엣지순 베스트 15 (배당 대비 가치, '확인 필요' 제외)", ""] + table(r["overall"]["edge"])
L += ["", "## 터치다운 확률 TOP 15", ""] + table(r["overall"]["td"])
L += ["", "## 경기별 베스트 15", ""]
for g in r["games"]:
    L += [f"### {g['away']} @ {g['home']} · {g['kickoff']} · {g['stadium']}",
          f"스프레드 {g['home']} {-g['spread_home']:+} · 총점 {g['total']} · 예상 점수 {g['away']} {g['proj']['away']} – {g['home']} {g['proj']['home']}", ""]
    L += table(g["best15_prob"]) + [""]
Path(f"output/week{r['week']}_report.md").write_text("\n".join(L), encoding="utf-8")
print(f"output/week{r['week']}_report.md")
