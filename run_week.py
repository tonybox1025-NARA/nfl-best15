"""이번 주 베스트 15 만들기.

  python run_week.py              # 전체 실행 (필요하면 모델 재학습)
  python run_week.py --refresh    # '최신 업데이트' 버튼: 부상·날씨·배당만 새로 받아 재계산
  python run_week.py --week 11    # 특정 주차
"""
import argparse

from nflbest import recommend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--force-odds", action="store_true", help="배당 캐시 무시 (크레딧 사용)")
    a = ap.parse_args()
    r = recommend.run(a.season, a.week, refresh_only=a.refresh, force_odds=a.force_odds)

    print(f"\n{r['season']} 시즌 {r['week']}주차 · {r['generated_at']} · {r['seconds']}초")
    for k, v in r["sources"].items():
        print(f"  [{k}] {v}")
    if r["changes"]:
        print("\n바뀐 점:")
        for c in r["changes"]:
            print("  " + c)
    print("\n=== 전체 베스트 15 (확률순) ===")
    for p in r["overall"]["prob"]:
        st = f" [{p['status']}]" if p["status"] else ""
        edge = f"  엣지 {p['edge']:+.1f}%p" if p.get("edge") is not None else ""
        print(f"{p['rank']:>2}. {p['name']:<22} {p['team']:>3} vs {p['opp']:<3} {p['label']:<18} {p['prob']:>5.1f}%{edge}{st}")
    print("\n=== TD 확률 TOP 15 ===")
    for p in r["overall"]["td"]:
        st = f" [{p['status']}]" if p["status"] else ""
        print(f"{p['rank']:>2}. {p['name']:<22} {p['team']:>3} vs {p['opp']:<3} {p['prob']:>5.1f}%{st}")


if __name__ == "__main__":
    main()
