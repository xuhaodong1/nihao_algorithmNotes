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

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
    }
}
