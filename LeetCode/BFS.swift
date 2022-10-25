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
}
