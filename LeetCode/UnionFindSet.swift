//
//  UnionFindSet.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/18.
//

import Foundation

/// 并查集等相关练习题
class UnionFindSet: BaseCode {

    /// 题目链接：[684. 冗余连接](https://leetcode.cn/problems/redundant-connection/)
    func findRedundantConnection(_ edges: [[Int]]) -> [Int] {
        let n = edges.count
        var parents = [Int](repeating: 0, count: n + 1)
        for i in 1...n {
            parents[i] = i
        }
        for edge in edges {
            if find(edge[0]) != find(edge[1]) {
                union(edge[0], edge[1])
            } else {
                return edge
            }
        }
        func union(_ index1: Int, _ index2: Int) {
            parents[find(index1)] = find(index2)
        }
        func find(_ index: Int) -> Int {
            var index = index
            while parents[index] != index {
                index = parents[index]
            }
            return index
        }
        return []
    }

    /// 题目链接：[827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)
    func largestIsland(_ grid: [[Int]]) -> Int {
        let n = grid.count
        var parent = [[(Int, Int)]](repeating: [(Int, Int)](repeating: (0, 0), count: n), count: n)
        var size = [[Int]](repeating: [Int](repeating: 1, count: n), count: n)
        let dirs = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        var ans = 0
        func union(_ p1: (Int, Int), _ p2: (Int, Int)) {
            let r1 = find(p1), r2 = find(p2)
            if r1 == r2 { return }
            parent[r2.0][r2.1] = r1
            size[r1.0][r1.1] += size[r2.0][r2.1]
        }
        func find(_ p: (Int, Int)) -> (Int, Int) {
            var ans = p
            while parent[ans.0][ans.1] != ans {
                ans = parent[ans.0][ans.1]
            }
            return ans
        }
        for i in 0..<n {
            for j in 0..<n {
                parent[i][j] = (i, j)
            }
        }
        for i in 0..<n {
            for j in 0..<n where grid[i][j] != 0 {
                for dir in dirs {
                    let x = i + dir[0], y = j + dir[1]
                    if x < 0 || y < 0 || x >= n || y >= n || grid[x][y] == 0 { continue }
                    union((i, j), (x, y))
                }
            }
        }
        for i in 0..<n {
            for j in 0..<n {
                if grid[i][j] == 1 {
                    let p = find((i, j))
                    ans = max(ans, size[p.0][p.1])
                } else {
                    var tot = 1
                    var arr = Set<Int>()
                    for dir in dirs {
                        let x = i + dir[0], y = j + dir[1]
                        if x < 0 || y < 0 || x >= n || y >= n || grid[x][y] == 0 { continue }
                        let p = find((x, y))
                        let t = p.0 * n + p.1 + 1
                        if arr.contains(t) { continue }
                        tot += size[p.0][p.1]
                        arr.insert(t)
                    }
                    ans = max(ans, tot)
                }
            }
        }
        return ans
    }

    override var excuteable: Bool {
        return true
    }

    override func executeTestCode() {
        super.executeTestCode()
        print(largestIsland([[1, 1], [1, 1]]))
    }
}
