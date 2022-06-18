//
//  ModelDesign.swift
//  LeetCode
//
//  Created by haodong xu on 2022/2/6.
//

import Foundation

class Bitset {
    private var bits: [Int]
    private var oneCount: Int
    private var zeroCount: Int
    private var isFlip: Bool
    init(_ size: Int) {
        bits = [Int].init(repeating: 0, count: size)
        oneCount = 0
        zeroCount = size
        isFlip = false
    }
    func fix(_ idx: Int) {
        let bit = isFlip ? bits[idx] ^ 1 : bits[idx]
        if bit != 1 {
            oneCount += 1
            zeroCount -= 1
            bits[idx] ^= 1
        }
    }
    func unfix(_ idx: Int) {
        let bit = isFlip ? bits[idx] ^ 1 : bits[idx]
        if bit != 0 {
            zeroCount += 1
            oneCount -= 1
            bits[idx] ^= 1
        }
    }
    func flip() {
        isFlip = !isFlip
        let temp = oneCount
        oneCount = zeroCount
        zeroCount = temp
    }
    func all() -> Bool {
        return oneCount == bits.count
    }
    func one() -> Bool {
        return oneCount > 0
    }
    func count() -> Int {
        return oneCount
    }
    func toString() -> String {
        var ans = ""
        for bit in bits {
            ans += "\(isFlip ? bit ^ 1 : bit)"
        }
        return ans
    }
}

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public var pre: ListNode?
    public init() { self.val = 0; self.next = nil; }
    public init(_ val: Int) { self.val = val; self.next = nil; }
    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}

public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init() { self.val = 0; self.left = nil; self.right = nil; }
    public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
    public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
        self.val = val
        self.left = left
        self.right = right
    }
}

//CustomerReflectable protocol.
public class Node {
    public var val: Int
    public var children: [Node]
    public init(_ val: Int) {
        self.val = val
        self.children = []
    }
}

public class SingalNode {
    public var val: Int
    public var next: SingalNode?
     public init(_ val: Int) {
         self.val = val
         self.next = nil
     }
 }

class LRUCache {
    var map: [Int: DListNode]
    var capacity: Int
    var head: DListNode?
    var tail: DListNode?

    init(_ capacity: Int) {
        map = [Int: DListNode]()
        self.capacity = capacity
        head = DListNode(0, 0) // 伪头节点
        tail = DListNode(0, 0) // 伪尾节点
        head?.next = tail
        tail?.pre = head
    }

    func get(_ key: Int) -> Int {
        let node = map[key]
        if node != nil {
            moveNodeToHead(node: node!)
        }
        return node?.val ?? -1
    }

    func put(_ key: Int, _ value: Int) {
        if map.keys.contains(key) {
            let node = map[key]!
            node.val = value
            moveNodeToHead(node: node)
        } else {
            let node = DListNode.init(key, value)
            map[key] = node
            addToHead(node: node)
            if map.count > capacity {
                if let node = removeTail() {
                    map.removeValue(forKey: node.key)
                }
            }
        }
    }

    private func addToHead(node: DListNode) {
        let next = head?.next
        head?.next = node
        node.next = next
        next?.pre = node
        node.pre = head
    }

    private func removeNode(node: DListNode) {
        let preNode = node.pre
        let nextNode = node.next
        preNode?.next = nextNode
        nextNode?.pre = preNode
    }

    private func moveNodeToHead(node: DListNode) {
        removeNode(node: node)
        addToHead(node: node)
    }

    private func removeTail() -> DListNode? {
        let prePreNode = tail?.pre?.pre
        let returnNode = prePreNode?.next
        prePreNode?.next = tail
        tail?.pre = prePreNode
        return returnNode
    }

    public class DListNode {
        public var key: Int
        public var val: Int
        public var next: DListNode?
        public var pre: DListNode?
        public init(_ key: Int, _ val: Int) {
            self.key = key
            self.val = val
        }
    }
}

class AllOne {
    var root: TNode
    var nodes: [String: TNode]
    init() {
        root = TNode.init(0)
        root.next = root
        root.pre = root
        nodes = [String: TNode]()
    }

    func inc(_ key: String) {
        if nodes.keys.contains(key) { // 在链表中
            let curr = nodes[key]!
            let next = curr.next
            if next == root || (next?.cnt ?? 0) > curr.cnt + 1 {
                let new = TNode.init(curr.cnt + 1, key: key)
                curr.insert(node: new)
                nodes[key] = new
            } else {
                next?.keys.insert(key)
                nodes[key] = next
            }
            curr.keys.remove(key)
            if curr.keys.isEmpty {
                curr.remove()
            }

        } else { // 不在链表中
            if root.next == root || root.next!.cnt > 1 {
                let node = TNode.init(1, key: key)
                root.insert(node: node)
                nodes[key] = node
            } else {
                root.next?.keys.insert(key)
                nodes[key] = root.next
            }
        }
    }

    func dec(_ key: String) {
        let curr = nodes[key]!
        if curr.cnt == 1 {
            nodes.removeValue(forKey: key)
        } else {
            let pre = curr.pre
            if pre == root || pre!.cnt < curr.cnt - 1 {
                let node = TNode.init(curr.cnt - 1, key: key)
                pre?.insert(node: node)
                nodes[key] = node
            } else {
                pre?.keys.insert(key)
                nodes[key] = pre
            }
        }
        curr.keys.remove(key)
        if curr.keys.isEmpty {
            curr.remove()
        }
    }

    func getMaxKey() -> String {
        return root.pre == root ? "" : (root.pre?.keys.randomElement() ?? "")
    }

    func getMinKey() -> String {
        return root.next == root ? "" : (root.next?.keys.randomElement() ?? "")
    }


    public class TNode: Equatable {
        public static func == (lhs: AllOne.TNode, rhs: AllOne.TNode) -> Bool {
            return lhs.cnt == rhs.cnt && lhs.keys == rhs.keys
        }

        var pre: TNode?
        var next: TNode?
        var keys: Set<String>
        var cnt: Int
        init(_ cnt: Int) {
            keys = Set<String>()
            self.cnt = cnt
        }

        init(_ cnt: Int, key: String) {
            keys = [key]
            self.cnt = cnt
        }

        func insert(node: TNode?) {
            node?.pre = self
            node?.next = self.next
            self.next = node
            node?.next?.pre = node
        }

        func remove() {
            self.pre?.next = self.next
            self.next?.pre = self.pre
        }
    }
}

class Bank {
    var accounts: [Int]
    let n: Int
    init(_ balance: [Int]) {
        accounts = balance
        n = balance.count
    }

    func transfer(_ account1: Int, _ account2: Int, _ money: Int) -> Bool {
        if account1 <= n && account2 <= n && accounts[account1 - 1] >= money {
            accounts[account1 - 1] -= money
            accounts[account2 - 1] += money
            return true
        }
        return false
    }

    func deposit(_ account: Int, _ money: Int) -> Bool {
        if account <= n {
            accounts[account - 1] += money
            return true
        }
        return false
    }

    func withdraw(_ account: Int, _ money: Int) -> Bool {
        if account <= n && accounts[account - 1] >= money {
            accounts[account - 1] -= money
            return true
        }
        return false
    }
}

class Pair {
    var key: Int
    var value: Int
    var next: Pair?
    var pre: Pair?
    init(key: Int, value: Int) {
        self.key = key
        self.value = value
    }
}

class MyHashMap {
    let base = 769
    var list: [Pair]
    init() {
        list = [Pair]()
        for _ in 0..<base {
            list.append(.init(key: -1, value: -1))
        }
    }

    func put(_ key: Int, _ value: Int) {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        if pair != nil {
            pair?.value = value
        } else {
            var tail: Pair? = list[i]
            while tail?.next != nil {
                tail = tail?.next
            }
            let new = Pair.init(key: key, value: value)
            new.pre = tail
            tail?.next = new
        }
    }

    func get(_ key: Int) -> Int {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        return pair?.value ?? -1
    }

    func remove(_ key: Int) {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        if let pair = pair {
            pair.pre?.next = pair.next
            pair.next = pair.pre
        }
    }

    private func hash(key: Int) -> Int {
        return key % base
    }
}

class Encrypter {
    var mapKeys: [Character: String]
    var mapValues: [String: [Character]]
    var set: [String: Int]
    init(_ keys: [Character], _ values: [String], _ dictionary: [String]) {
        mapKeys = [Character: String]()
        mapValues = [String: [Character]]()
        for i in 0..<keys.count {
            let key = keys[i], value = values[i]
            mapKeys[key] = value
            if mapValues[value] == nil {
                mapValues[value] = [key]
            } else {
                mapValues[value]?.append(key)
            }
        }
        set = [String: Int]()
        for str in dictionary {
            var ans = ""
            for c in str {
                ans += mapKeys[c]!
            }
            set[ans] = (set[ans] ?? 0) + 1
        }
    }

    func encrypt(_ word1: String) -> String {
        return word1.reduce("", { $0 + mapKeys[$1]! })
    }

    func decrypt(_ word2: String) -> Int {
        return set[word2] ?? 0
    }
}

class SegmentTreeNumArray {
    class Node {
        var v: Int = -1
        var l = -1, r = -1
        init() {}
        init(v: Int, l: Int, r: Int) {
            self.v = v
            self.l = l
            self.r = r
        }
        init(l: Int, r: Int) {
            self.l = l
            self.r = r
        }
    }
    var tree: [Node]
    let n: Int
    var nums: [Int]

    init(_ nums: [Int]) {
        n = nums.count
        self.nums = nums
        tree = [Node].init(repeating: Node(), count: n << 2)
        buildTree(cur: 1, l: 1, r: n)
    }

    @discardableResult
    private func buildTree(cur: Int, l: Int, r: Int) -> Node {
        let node = Node(l: l, r: r)
        if l == r {
            node.v = nums[l - 1]
        } else {
            let mid = (l + r) >> 1
            let left = buildTree(cur: cur << 1, l: l, r: mid)
            let right = buildTree(cur: cur << 1 | 1, l: mid + 1, r: r)
            node.v = left.v + right.v
        }
        tree[cur] = node
        return node
    }

    private func update(cur: Int, index: Int, v: Int) {
        if tree[cur].l == index && tree[cur].r == index {
            tree[cur].v = v
            return
        }
        let mid = (tree[cur].l + tree[cur].r) >> 1
        if index <= mid {
            update(cur: cur << 1, index: index, v: v)
        } else {
            update(cur: cur << 1 | 1, index: index, v: v)
        }
        tree[cur].v = tree[cur << 1].v + tree[cur << 1 | 1].v
    }

    private func query(cur: Int, l: Int, r: Int) -> Int {
        if tree[cur].l >= l && tree[cur].r <= r {
            return tree[cur].v
        }
        var sum = 0
        let mid = (tree[cur].l + tree[cur].r) >> 1
        if mid >= l {
            sum += query(cur: cur << 1, l: l, r: r)
        }
        if r > mid {
            sum += query(cur: cur << 1 | 1, l: l, r: r)
        }
        return sum
    }

    func update(_ index: Int, _ val: Int) {
        update(cur: 1, index: index + 1, v: val)
        nums[index] = val
    }

    func sumRange(_ left: Int, _ right: Int) -> Int {
        query(cur: 1, l: left + 1, r: right + 1)
    }
}

class NumArray {
    var tree: [Int]
    var nums: [Int]
    let n: Int
    init(_ nums: [Int]) {
        n = nums.count
        tree = [Int].init(repeating: 0, count: n + 1)
        self.nums = nums
        for i in 0..<n {
            add(i + 1, nums[i])
        }
    }

    func update(_ index: Int, _ val: Int) {
        add(index + 1, val - nums[index])
        nums[index] = val
    }

    func add(_ index: Int, _ val: Int) {
        var index = index
        while index <= n {
            tree[index] += val
            index += lowBit(x: index)
        }
    }

    func query(_ index: Int) -> Int {
        var index = index + 1, ans = 0
        while index > 0 {
            ans += tree[index]
            index -= lowBit(x: index)
        }
        return ans
    }

    func sumRange(_ left: Int, _ right: Int) -> Int {
        return query(right) - query(left - 1)
    }

    private func lowBit(x: Int) -> Int {
        return x & -x
    }
}

class RandomizedSet {
    var set: Set<Int>
    init() {
        set = Set<Int>()
    }

    func insert(_ val: Int) -> Bool {
        if set.contains(val) {
            return false
        }
        set.insert(val)
        return true
    }

    func remove(_ val: Int) -> Bool {
        if !set.contains(val) {
            return false
        }
        set.remove(val)
        return true
    }

    func getRandom() -> Int {
        set.randomElement() ?? -1
    }
}

class NestedInteger {
    var value: Int
    var list = [NestedInteger]()
    init(_ value: Int) {
        self.value = value
    }
    public func isInteger() -> Bool {
        return list.isEmpty
    }
    public func getInteger() -> Int {
        return value
    }
    public func setInteger(value: Int) {
        self.value = value
    }
    public func add(elem: NestedInteger) {
        list.append(elem)
    }
    public func getList() -> [NestedInteger] {
        return list
    }
}

class RecentCounter {
    var queue: [Int]
    init() {
        queue = [Int]()
    }

    func ping(_ t: Int) -> Int {
        let pre = t - 3000
        while !queue.isEmpty && queue.first! < pre { queue.removeFirst() }
        queue.append(t)
        return queue.count
    }
}

class CountIntervals {
    var arr: [[Int]]
    init() {
        arr = [[Int]]()
    }

    func add(_ left: Int, _ right: Int) {
        if arr.isEmpty {
            arr.append([left, right])
            return
        }
        var l = 0, r = arr.count
        while l < r {
            let mid = (l + r) >> 1
            if arr[mid][0] > left {
                r = mid
            } else {
                l = mid + 1
            }
        }
        arr.insert([left, right], at: l)
    }

    func count() -> Int {
        if arr.isEmpty {
            return 0
        }
        var curr = arr[0][1] + 1
        var ans = arr[0][1] - arr[0][0] + 1
        for i in 1..<arr.count {
            if arr[i][1] < curr {
                continue
            }
            if arr[i][0] >= curr {
                ans += arr[i][1] - arr[i][0] + 1
            } else {
                ans += arr[i][1] - curr + 1
            }
            curr = arr[i][1] + 1
        }
        return ans
    }

    func inorderSuccessor(_ root: TreeNode?, _ p: TreeNode?) -> TreeNode? {
        guard let root = root, let p = p else {
            return nil
        }
        var queue: [TreeNode] = [root], curr: TreeNode? = root
        var ans: TreeNode?
        while !queue.isEmpty || curr != nil {
            while curr?.left != nil {
                queue.append(curr!.left!)
                curr = curr?.left!
            }
            let node = queue.removeLast()
            if node.val < p.val {
                ans = node
                break
            }
            curr = curr?.right
        }
        return ans
    }
}

class MinStack {
    var stack: [Int]
    var nums: [Int]

    init() {
        stack = [Int]()
        nums = [Int]()
    }

    func push(_ val: Int) {
        if stack.isEmpty {
            stack.append(val)
        } else {
            stack.append(stack.last! < val ? stack.last! : val)
        }
        nums.append(val)
    }

    func pop() {
        stack.removeLast()
        nums.removeLast()
    }

    func top() -> Int {
        return nums.last!
    }

    func getMin() -> Int {
        return stack.last!
    }
}

class Solution2 {
    var radius: Double
    var xCenter: Double
    var yCenter: Double
    init(_ radius: Double, _ x_center: Double, _ y_center: Double) {
        self.radius = radius
        self.xCenter = x_center
        self.yCenter = y_center
    }

    func randPoint() -> [Double] {
        var x: Double = 0.0, y: Double = 0.0
        repeat {
            x = Double.random(in: -radius...radius)
            y = Double.random(in: -radius...radius)
        } while x * x + y * y > radius * radius
        return [x + xCenter, y + yCenter]
    }
}

class TextEditor {
    var preText: [Character]
    var tailText: [Character]
    init() {
        preText = [Character]()
        tailText = [Character]()
    }

    func addText(_ text: String) {
        for char in text {
            preText.append(char)
        }
    }

    func deleteText(_ k: Int) -> Int {
        let cnt = preText.count
        if k > cnt {
            preText.removeAll()
            return cnt
        } else {
            preText.removeSubrange(cnt-k..<cnt)
            return k
        }
    }

    func cursorLeft(_ k: Int) -> String {
        for _ in 0..<k {
            if preText.count > 0 {
                tailText.append(preText.removeLast())
            } else {
                break
            }
        }
        var ans = ""
        let n = preText.count
        for i in (0..<min(n, 10)).reversed() {
            ans.append(preText[n - i - 1])
        }
        return ans
    }

    func cursorRight(_ k: Int) -> String {
        for _ in 0..<k {
            if tailText.count > 0 {
                preText.append(tailText.removeLast())
            } else {
                break
            }
        }
        var ans = ""
        let n = preText.count
        for i in (0..<min(n, 10)).reversed() {
            ans.append(preText[n - i - 1])
        }
        return ans
    }
}

class MyCalendarThree {
    var map: [Int: Int]
    init() {
        map = [Int: Int]()
    }

    func book(_ start: Int, _ end: Int) -> Int {
        var ans = 0
        var maxBook = 0
        map[start] = (map[start] ?? 0) + 1
        map[end] = (map[end] ?? 0) - 1
        for item in map.sorted(by: { item1, item2 in
            return item1.key < item2.key
        }) {
            let freq = item.value
            maxBook += freq
            ans = max(maxBook, ans)
        }
        return ans
    }
}

class Solution3 {
    var rects: [[Int]]
    var preSum: [Int]

    init(_ rects: [[Int]]) {
        self.rects = rects
        let n = rects.count
        preSum = [Int].init(repeating: 0, count: n + 1)
        for i in 1...n {
            preSum[i] = preSum[i - 1] + (rects[i - 1][2] - rects[i - 1][0] + 1) * (rects[i - 1][3] - rects[i - 1][1] + 1)
        }
    }

    func pick() -> [Int] {
        let select = Int.random(in: 1...preSum.max()!)
        var l = 1, r = rects.count
        while l < r {
            let mid = l + (r - l) >> 1
            if preSum[mid] + 1 >= select {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return [Int.random(in: rects[l - 1][0]...rects[l - 1][2]), Int.random(in: rects[l - 1][1]...rects[l - 1][3])]
    }
}
