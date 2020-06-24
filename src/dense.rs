use crate::VoxelNode;
use crate::VoxelTree;
use crate::allocate;
use std::ops::IndexMut;

pub struct DenseVoxelData {
    pub size : usize,
    depth : usize,
    pub data : Vec<VoxelNode>,
}

impl DenseVoxelData {// the dimensions are actually 2^n
    pub fn new(size : u32) -> Self {
        DenseVoxelData {
            size: 2usize.pow(size),
            depth : size as usize,
            data : vec![VoxelNode::empty(); 2usize.pow(size).pow(3u32)]
        }
    }

    pub fn access(&mut self, x : usize, y : usize, z : usize) -> &mut VoxelNode {
        debug_assert!(x < self.size, "AAh, out of range.");
        debug_assert!(y < self.size, "AAh, out of range.");
        debug_assert!(z < self.size, "AAh, out of range.");
        self.data.index_mut(x + (y * self.size) + (z * (self.size * self.size)))
    }

    pub fn to_sparse(&self) -> VoxelTree {
        let mut tree = allocate(self.size.pow(3) * 3); //3 is the size per node in bytes. at the time of writing
        let rootid = tree.allocate_node();
        let root: &mut VoxelNode = tree.get_node(rootid);

        let mut levels: Vec<Vec<u32>> = Vec::new();

        levels.push(vec![rootid]);

        for x in 0..self.depth{
            levels.push (Vec::new());
            levels[x + 1].reserve(8usize.pow((x + 1) as u32));
        }

        fn do_child<'a>(level : usize, node : u32, tree : VoxelTree, levels : &mut Vec<Vec<u32>>){
            if level >= levels.len() {
                return;
            }
            tree.get_node(node).childmask= u8::MAX;
            for x in 0..8 {
                tree.get_node(node).put_child(x, tree.allocate_node());
                let child = tree.get_node(node).get_child_ptr(x);

                levels[level].push(child);
                do_child(level + 1, child, tree, levels);
            }
        }

        do_child(0, rootid, tree, &mut levels);

        tree
    }
}
