# 최단경로
- 가장 짧은 경로를 찾는 알고리즘 (a.k.a 길 찾기 문제)
- 그래프를 이용해 표현할 수 있다. -> 그림 그려보기!
    - 노드 : 각 지점
    - 간선 : 각 지점 사이에 연결된 도로

## 1. 다익스트라 최단 경로 알고리즘
```
D[i][j][k] = str[a] # 닉값 제대로 하시는 분...
```
- **한** 지점에서 다른 **특정** 지점까지의 최단 경로를 구해야 하는 경우

## 2. 플로이드 워셜
- **모든** 지점에서 다른 **모든** 지점까지의 최단 경로를 모두 구해야 하는 경우
## 3. 벨만 포드 알고리즘

### 실전문제 3번 : 전보 (p.262)

내 코드
```
n, m, c = map(int, input().split()) # 도시 갯수 n, 통로 갯수 m, 전보 보내는 도시 c
INF = int(1e9)
cnt = 0 # 도시 c에서 전보 보낼 수 있는 도시의 총 갯수
max_time = 0 # 최대 시간

graph = [[INF]*(n+1) for _ in range(n+1)]

for a in range(1, n+1):
  for b in range(1, n+1):
    if a == b:
      graph[a][b] = 0

for _ in range(m):
  x, y, z = map(int, input().split())
  # x 에서 y 까지 걸리는 시간 z
  graph[x][y] = z

for k in range(1, n+1):
  for a in range(1, n+1):
    for b in range(1, n+1):
      graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

for b in range(1, n+1):
  if graph[c][b] < INF and b != c:
    cnt = cnt +1
    max_time = max(max_time, graph[c][b])

print(cnt, max_time)

```

결과
```
>>> 3 2 1
>>> 1 2 4
>>> 1 3 2
2 4
```

💡 설명
- 플로이드-워셜로 풀었음
- INF 아니고 0 (제자리) 아니면 연결된 거니까, cnt 증가시켜줌
- graph 테이블 중에 들어있는 값들 중 가장 큰 값 (INF 제외)이 max_time

🥅 개선할 점
- 도시 C에서 출발하는 것만 보면 되니까, **한** 도시에서 다른 도시까지의 최단 거리 문제로 치환할 수 있다. 
- 따라서 플로이드-워셜이 아닌 다익스트라 알고리즘으로 풀 수도 있다. (나는 플로이드-워셜 배운 직후라 써먹고 싶어서 이걸로 풀었다!)
- N과 M의 범위가 충분히 크기 때문에, 다익스트라로 풀려면 우선순위 큐를 사용해야한다.
- input 받을 때도 readline 으로 받는 게 실행시간 측면에서 더 유리하다. (이건 그냥 해도 딱히 문제 없어서 계속 안쓰게 되는듯...! 개선할 때는 꼭 readline으로 받자)

⚡️ 개선된 코드
```

```