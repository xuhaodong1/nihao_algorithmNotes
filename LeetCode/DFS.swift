//
//  DFS.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/2.
//

import Foundation

/// DFS 相关练习题
class DeepFirstSearch: BaseCode {

    /// 题目链接：[687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)
    func longestUnivaluePath(_ root: TreeNode?) -> Int {
        var ans = 0
        @discardableResult
        func dfs(node: TreeNode?) -> Int {
            guard let node = node else { return 0 }
            let leftCnt = dfs(node: node.left)
            let rightCnt = dfs(node: node.right)
            var currLeft = 0, currRight = 0
            if let left = node.left, left.val == node.val {
                currLeft = 1 + leftCnt
            }
            if let right = node.right, right.val == node.val {
                currRight += 1 + rightCnt
            }
            ans = max(ans, currLeft + currRight)
            return max(currLeft, currRight)
        }
        dfs(node: root)
        return ans
    }

    /// 题目链接：[652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/submissions/)
    func findDuplicateSubtrees(_ root: TreeNode?) -> [TreeNode?] {
        guard let root = root else { return [] }
        var map = [String: Int]()
        var ans = [TreeNode]()
        @discardableResult
        func dfs(node: TreeNode?) -> String {
            guard let node = node else { return "nil" }
            var id = "\(node.val)"
            id += "_\(dfs(node: node.left))"
            id += "_\(dfs(node: node.right))"
            if map[id] == 1 { ans.append(node) }
            map[id, default: 0] += 1
            return id
        }
        dfs(node: root)
        return ans
    }

    /// 题目链接：[669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)
    func trimBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> TreeNode? {
        func dfs(_ node: TreeNode?) -> TreeNode? {
            guard let node = node else { return nil }
            if node.val < low {
                return dfs(node.right)
            } else if node.val > high {
                return dfs(node.left)
            }
            node.left = dfs(node.left)
            node.right = dfs(node.right)
            return node
        }
        return dfs(root)
    }

    /// 题目链接：[698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)
    func canPartitionKSubsets(_ nums: [Int], _ k: Int) -> Bool {
        let sum = nums.reduce(0, +), n = nums.count
        let nums = nums.sorted(by: >), subSum = sum / k
        if sum % k != 0 { return false } // 前置性判断
        if nums.contains(where: { $0 > subSum }) { return false } // 前置性判断
        var visited = [Bool](repeating: false, count: n)
        func dfs(currSum: Int, cnt: Int, idx: Int) -> Bool {
            if cnt == k { return true }
            if currSum == subSum { return dfs(currSum: 0, cnt: cnt + 1, idx: 0) } // 新一轮 idx 改为 0
            for i in idx..<n where !visited[i] && currSum + nums[i] <= subSum { // 从前往后搜, 剪枝
                visited[i] = true
                if dfs(currSum: currSum + nums[i], cnt: cnt, idx: i + 1) { return true }
                visited[i] = false
                if currSum == 0 { return false } // 如果没有与其匹配的, 不用计算后面的
            }
            return false
        }
        return dfs(currSum: 0, cnt: 0, idx: 0)
    }

    /// 题目链接：[854. 相似度为 K 的字符串](https://leetcode.cn/problems/k-similar-strings/)
    func kSimilarity(_ s1: String, _ s2: String) -> Int {
        let n = s1.count
        var chars1 = [Character](s1), chars2 = [Character](s2)
        var ans = Int.max
        func dfs(currIndex: Int, currK: Int) {
            if currK >= ans { return }
            if currIndex == n - 1 {
                ans = min(ans, currK)
                return
            }
            if chars1[currIndex] == chars2[currIndex] {
                dfs(currIndex: currIndex + 1, currK: currK)
            } else {
                for j in currIndex+1..<n where chars1[j] == chars2[currIndex] && chars1[j] != chars2[j] {
                    chars1.swapAt(currIndex, j)
                    dfs(currIndex: currIndex + 1, currK: currK + 1)
                    chars1.swapAt(currIndex, j)
                }
            }
        }
        dfs(currIndex: 0, currK: 0)
        return ans
    }

    override var excuteable: Bool {
        return true
    }

    override func executeTestCode() {
        super.executeTestCode()
        print(kSimilarity("abc", "bca"))
    }
}
