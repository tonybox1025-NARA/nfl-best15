# NFL 베스트 15

매주 NFL 선수 프롭(TD, 러싱·리시빙 야드, 리셉션, 패싱)의 확률을 계산해서
**경기별 베스트 15**와 **전체 경기 베스트 15**를 1위부터 순서대로 뽑아요.

> 확률은 과거 데이터 기반 추정치예요. 반드시 맞는다는 뜻이 아니에요.

## 실행

```bash
pip install -r requirements.txt
python run_week.py              # 이번 주 전체 계산 (모델은 하루 한 번 재학습)
python run_week.py --refresh    # 최신 업데이트: 부상·날씨·배당만 새로 받아 재계산
python backtest.py              # 2025 시즌 검증 (TEST=2024 python backtest.py 도 가능)
```

결과: `output/latest.json` (앱이 읽는 파일), `output/week{N}.json`

## 데이터 (모두 무료)

| 데이터 | 출처 | 비고 |
|---|---|---|
| 일정·스프레드·총점·경기장 | nflverse (nflreadpy) | |
| 선수 경기 기록 2020~현재 | nflverse | |
| 레드존(10야드 안쪽) 캐리·타깃 | nflverse play-by-play | TD 예측 핵심 |
| 부상 리포트·훈련 참여 | nflverse + ESPN 실시간 | |
| 날씨 (킥오프 시각) | Open-Meteo | 돔 경기는 제외 |
| FanDuel 선수 프롭 배당 | The Odds API | `ODDS_API_KEY` 필요, 무료 월 500크레딧 |

배당 크레딧 계산: 1경기 조회 = 마켓 수만큼. 기본 3개 마켓 × 14경기 ≈ 42크레딧/회 → 무료로는 한 달에 약 11회.
`ODDS_CACHE_MIN`(기본 10분) 안에 다시 누르면 이전 배당을 재사용해요.

## 경기 직전 비출전 명단 수동 반영

`overrides.json`:
```json
{"out": ["James Cook"], "active": ["Puka Nacua"]}
```

## 모델 요약

- 피처: 최근 3/8경기·가중 평균 사용량(캐리·타깃·점유율·레드존), 상대 수비 포지션별 허용치(최근 8경기),
  팀 공격 성향, 팀 예상 득점(스프레드·총점), 홈/원정, 돔, 바람
- TD: LightGBM 분류 + 확률 보정
- 야드/리셉션: LightGBM로 예상값 → 비슷한 예상값 선수들의 실제 분포로 'N 이상 확률' → 2단계 보정
- 결장 선수의 타깃/캐리는 같은 팀 동료에게 70% 재분배
- 베스트 15: 선수당 최대 2개, 같은 종류 마켓 최대 4개

## 폴더

```
nflbest/data.py       nflverse 로딩·캐시
nflbest/features.py   경기 전 피처 계산
nflbest/model.py      학습·예측·확률 보정
nflbest/markets.py    마켓·기준선 설정
nflbest/picks.py      후보 픽·베스트 15
nflbest/live.py       부상·날씨·배당 실시간
nflbest/recommend.py  주차 파이프라인 → JSON
.github/workflows/    자동 실행 + 업데이트 버튼용
```
