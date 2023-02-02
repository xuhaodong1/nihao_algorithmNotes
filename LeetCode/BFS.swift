//
//  BFS.swift
//  LeetCode
//
//  Created by haodong xu on 2022/10/25.
//

import Foundation

/// BFS 相关练习题
class BFS: BaseCode {

    /// 题目链接：[934. 最短的桥](https://leetcode.cn/problems/shortest-bridge/)
    func shortestBridge(_ grid: [[Int]]) -> Int {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        var grid = grid
        var queue = [(Int, Int)]()
        let n = grid.count
        for i in 0..<n {
            for j in 0..<n where grid[i][j] == 1 {
                dfs(i, j)
                break
            }
            if !queue.isEmpty { break }
        }
        func dfs(_ i: Int, _ j: Int) {
            grid[i][j] = 2
            queue.append((i, j))
            for dir in dirs {
                let x = i + dir.0, y = j + dir.1
                if isValid(x: x, y: y) && grid[x][y] != 0 && grid[x][y] != 2 { dfs(x, y) }
            }
        }
        func isValid(x: Int, y: Int) -> Bool { return x >= 0 && x < n && y >= 0 && y < n }
        var ans = 0
        while !queue.isEmpty {
            let cnt = queue.count
            for _ in 0..<cnt {
                let curr = queue.removeFirst()
                for dir in dirs {
                    let x = curr.0 + dir.0, y = curr.1 + dir.1
                    if !isValid(x: x, y: y) { continue }
                    if grid[x][y] == 1 { return ans }
                    else if grid[x][y] == 0 {
                        queue.append((x, y))
                        grid[x][y] = 2
                    }
                }
            }
            ans += 1
        }
        return ans
    }

    /// 题目链接：[864. 获取所有钥匙的最短路径](https://leetcode.cn/problems/shortest-path-to-get-all-keys/description/)
    /// BFS + 状态压缩
    func shortestPathAllKeys(_ grid: [String]) -> Int {
        let grid = grid.map{ [Character]($0) }
        let n = grid.count, m = grid[0].count, INF = 0x3f3f3f3f
        let dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        var queue = [(Int, Int, Int)]()
        // dist[x][y][state] 表示 在 x,y 坐标下的 state 状态下所需要的步数
        var dist = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: INF, count: 1 << 6), count: m), count: n)
        var cnt = 0
        for i in 0..<n {
            for j in 0..<m {
                let c = grid[i][j]
                if c == "@" {
                    dist[i][j][0] = 0
                    queue.append((i, j, 0))
                } else if c >= "a" && c <= "z" { cnt += 1 }
            }
        }
        while !queue.isEmpty {
            let info = queue.removeFirst()
            let x = info.0, y = info.1, cur = info.2, step = dist[x][y][cur]
            for dir in dirs {
                let nx = x + dir[0], ny = y + dir[1]
                if nx < 0 || ny < 0 || nx >= n || ny >= m { continue }
                let c = grid[nx][ny]
                if c == "#" { continue } // 墙
                else if c >= "A" && c <= "Z" && (cur >> Int(c.asciiValue! - Character("A").asciiValue!)) & 1 == 0 { continue } // 没有钥匙
                var ncur = cur
                if c >= "a" && c <= "z" { ncur |= (1 << Int(c.asciiValue! - Character("a").asciiValue!)) } // 挂上钥匙
                if ncur == ((1 << cnt) - 1) { return step + 1 } // 满足所有钥匙
                if step + 1 >= dist[nx][ny][ncur] { continue } // 当前状态下步数比之前来过的大
                dist[nx][ny][ncur] = step + 1
                queue.append((nx, ny, ncur))
            }
        }
        return -1
    }

    /// 题目链接：[1129. 颜色交替的最短路径](https://leetcode.cn/problems/shortest-path-with-alternating-colors/)
    func shortestAlternatingPaths(_ n: Int, _ redEdges: [[Int]], _ blueEdges: [[Int]]) -> [Int] {
        var ans = [Int](repeating: 0, count: n)
        var g = [[[Int]]](repeating: [[Int]](repeating: [], count: n), count: 2)
        for redEdge in redEdges { g[0][redEdge[0]].append(redEdge[1]) }
        for blueEdge in blueEdges { g[1][blueEdge[0]].append(blueEdge[1]) }
        var dist = [[Int]](repeating: [Int](repeating: Int.max, count: n), count: 2)
        dist[0][0] = 0; dist[1][0] = 0
        var queue = [(Int, Int)]()
        queue.append((1, 0)); queue.append((0, 0))
        while !queue.isEmpty {
            let pair = queue.removeFirst()
            let t = pair.0, x = pair.1
            for y in g[1 ^ t][x] where dist[1 ^ t][y] == Int.max {
                dist[1 ^ t][y] = dist[t][x] + 1
                queue.append((1 ^ t, y))
            }
        }
        for i in 0..<n {
            ans[i] = min(dist[0][i], dist[1][i])
            if ans[i] == Int.max {
                ans[i] = -1
            }
        }
        return ans
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(shortestAlternatingPaths(3, [[0,1]], [[1,2]]))
    }
}
