use crate::allocate;
use crate::clamp;
use crate::oct_byte_to_vec;
use crate::VoxelNode;
use crate::VoxelTree;
use cgmath::Vector3;
use std::ops::Deref;
use std::ops::Index;
use std::ops::IndexMut;

pub struct DenseVoxelData {
    pub size: usize,
    depth: usize,
    pub data: Vec<VoxelNode>,
}

impl DenseVoxelData {
    // the dimensions are actually 2^n
    pub fn new(depth: u32) -> Self {
        DenseVoxelData {
            size: 2usize.pow(depth),
            depth: depth as usize + 1,
            data: vec![VoxelNode::empty(); 2usize.pow(depth).pow(3u32)],
        }
    }

    #[inline(always)]
    pub fn access_mut(&mut self, x: usize, y: usize, z: usize) -> &mut VoxelNode {
        debug_assert!(x < self.size, "AAh, out of range.");
        debug_assert!(y < self.size, "AAh, out of range.");
        debug_assert!(z < self.size, "AAh, out of range.");
        self.data
            .index_mut(x + (y * self.size) + (z * (self.size * self.size)))
    }

    #[inline(always)]
    pub fn access(&self, x: usize, y: usize, z: usize) -> &VoxelNode {
        debug_assert!(x < self.size, "AAh, out of range.");
        debug_assert!(y < self.size, "AAh, out of range.");
        debug_assert!(z < self.size, "AAh, out of range.");
        self.data
            .index(x + (y * self.size) + (z * (self.size * self.size)))
    }

    pub fn to_sparse(&mut self) -> VoxelTree {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut count = 0;
        let mut current = 1;

        while current <= self.size {
            count += current.pow(3);
            current *= 2;
        }
        let mut tree = allocate((count * 10 + 1) * 4); //10 is the size per node in uints. at the time of writing
        
        let rootid = tree.allocate_node();
        let root: &mut VoxelNode = tree.get_node(rootid);
        root.flags = 0;
        root.childmask = 0;

        #[derive(Copy, Clone)]
        struct SparseNode {
            node: u32,
            child_hashes: [u64; 8],
        }
        impl SparseNode {
            fn hash(&self, tree: VoxelTree) -> u64 {
                let mut hasher = DefaultHasher::new();
                tree.get_node(self.node).hash(&mut hasher);
                self.child_hashes.hash(&mut hasher);
                hasher.finish()
            }
        }

        struct SparseNodeCont {
            sp: SparseNode,
            parent_index: u32,
            parent_child_index: u8,
            hash: u64,
        }

        let mut levels: Vec<Vec<SparseNodeCont>> = Vec::new();

        levels.push(Vec::new());
        levels[0].reserve(1);

        for x in 1..self.depth {
            levels.push(Vec::new());
            levels[x].reserve(8usize.pow((x + 1) as u32));
        }

        fn do_child<'a>(
            s: &DenseVoxelData,
            level: usize,
            node: u32,
            tree: VoxelTree,
            levels: &mut Vec<Vec<SparseNodeCont>>,
            parent_index: u32,
            parent_child_index: u8,
            size: usize,
            x: usize,
            y: usize,
            z: usize,
        ) -> u64 {
            if level >= levels.len() - 1 {
                let ac = s.access(x, y, z);
                if ac.flags & 1 == 1 {
                    *tree.get_node(node) = *ac;
                    tree.get_node(node).childmask = 0;
                    let sp = SparseNode {
                        node,
                        child_hashes: [0; 8],
                    };
                    let hash = sp.hash(tree);
                    levels[level].push(SparseNodeCont {
                        sp,
                        parent_index,
                        parent_child_index,
                        hash,
                    });
                    return hash;
                }
                return 0;
            }

            tree.get_node(node).flags = 0;
            tree.get_node(node).childmask = 0;
            let mut child_hashes = [0u64; 8];

            let levelindex = levels[level].len() as u32;

            for i in 0..8 {
                let childptr = tree.allocate_node();
                tree.get_node(node).put_child(i, childptr);

                let mut loc: Vector3<f64> = oct_byte_to_vec(1u8 << i);
                loc *= 2.0;

                child_hashes[i as usize] = do_child(
                    s,
                    level + 1,
                    childptr,
                    tree,
                    levels,
                    levelindex,
                    i,
                    size / 2,
                    (x as usize + (clamp(loc.x, 0.0, 1.0) as usize * size as usize / 2)) as usize,
                    (y as usize + (clamp(loc.y, 0.0, 1.0) as usize * size as usize / 2)) as usize,
                    (z as usize + (clamp(loc.z, 0.0, 1.0) as usize * size as usize / 2)) as usize,
                );
            }

            let sp = SparseNode { node, child_hashes };
            let hash = sp.hash(tree);
            levels[level].push(SparseNodeCont {
                sp,
                parent_index,
                parent_child_index,
                hash,
            });
            return hash;
        }

        do_child(
            &self,
            0,
            rootid,
            tree,
            &mut levels,
            0,
            0,
            self.size,
            0,
            0,
            0,
        );

        let mut tree2 = allocate((count * 10 + 1) * 4);

        let rootid2 = tree2.allocate_node();

        use std::collections::HashMap;

        let mut hm : HashMap<u32, (u32, bool)>= HashMap::new();

        let levelslen = levels.len();
        for leveli in (1..levels.len()).rev() {
            levels[leveli].sort_by(|a, b| a.hash.cmp(&b.hash));

            let mut last_hash = levels[leveli][0].hash;
            let mut replace = levels[leveli][0].sp.node;
            for i in 1..levels[leveli].len() {
                if levels[leveli][i].hash == last_hash {
                    let parent_index = levels[leveli][i].parent_index;
                    let parent_child_index = levels[leveli][i].parent_child_index;

                    let n: &mut VoxelNode =
                        tree.get_node(levels[leveli - 1][parent_index as usize].sp.node);

                    n.put_child(parent_child_index, replace);

                } else {
                    last_hash = levels[leveli][i].hash;
                    replace = levels[leveli][i].sp.node;
                    hm.insert(replace, (tree2.allocate_node(),false));
                }
            }
        }

        fn look_node(node: u32, tree: VoxelTree, tree_old: VoxelTree, hm : &mut HashMap<u32,(u32,bool)>) -> u32 {
            let (new_node_ptr, d) = match hm.get(&node) {
                Some(x) => {
                    *x
                },
                None => {
                    let allocation = tree.allocate_node();
                    hm.insert(node, (allocation, false));
                    (allocation,false)
                }
            };
            if d {
                return new_node_ptr;
            }

            let node_ref = tree_old.get_node(node);
            let ptr = tree.get_node(new_node_ptr);
            *ptr = *node_ref;
            ptr.childmask = 0;

            hm.insert(node, (new_node_ptr, true));

            for x in 0..8 {
                let child = 1 << x;
                if node_ref.childmask & child != 0 {
                    let child_node = look_node(node_ref.get_child_ptr(x), tree, tree_old, hm);
                    ptr.put_child(x, child_node);
                }
            }
            new_node_ptr
        }

        let node_ref = tree.get_node(rootid);
        let ptr = tree2.get_node(rootid2);
        for x in 0..8 {
            let child = 1 << x;
            if node_ref.childmask & child != 0 {
                let child_node = look_node(node_ref.get_child_ptr(x), tree2, tree, &mut hm);
                ptr.put_child(x, child_node);
            }
        }

        tree.free();

        tree2.reallocate_to_fit();

        tree2
    }
}
