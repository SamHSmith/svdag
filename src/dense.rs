use crate::allocate;
use crate::oct_byte_to_vec;
use crate::clamp;
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

    pub fn to_sparse(&self) -> VoxelTree {
        let mut count = 0;
        let mut current = 1;

        while current <= self.size {
            count += current.pow(3);
            current *= 2;
        }
        let mut tree = allocate((count * 10 + 1) * 4); //10 is the size per node in uints. at the time of writing
        println!("{}", (count * 10 + 1));
        let rootid = tree.allocate_node();
        let root: &mut VoxelNode = tree.get_node(rootid);
        root.flags = 0;
        root.childmask = 0;

        let mut levels: Vec<Vec<u32>> = Vec::new();

        levels.push(vec![rootid]);

        for x in 1..self.depth {
            levels.push(Vec::new());
            levels[x].reserve(8usize.pow((x + 1) as u32));
        }

        fn do_child<'a>(
            s: &DenseVoxelData,
            level: usize,
            node: u32,
            tree: VoxelTree,
            levels: &mut Vec<Vec<u32>>,
            size: usize,
            x: usize,
            y: usize,
            z: usize,
        ) {
            levels[level].push(node);
            if level >= levels.len() - 1 {
                println!("leaf {} {} {}, {}",x,y,z,s.access(x,y,z).emission);
                *tree.get_node(node) = *(s.access(x, y, z));
                tree.get_node(node).childmask = 0;
                println!("{} {}",node, tree.get_node(node).emission);
                return;
            }

            tree.get_node(node).flags = 0;
            tree.get_node(node).childmask = 0;
            for i in 0..8 {
                let childptr = tree.allocate_node();
                println!("{} :: {}", node, childptr);
                tree.get_node(node).put_child(i, childptr);

                let mut loc: Vector3<f64> = oct_byte_to_vec(1u8 << i);
                loc *= 2.0;

                do_child(
                    s,
                    level + 1,
                    childptr,
                    tree,
                    levels,
                    size / 2,
                    (x as usize + (clamp(loc.x,0.0, 1.0) as usize * size as usize / 2)) as usize,
                    (y as usize + (clamp(loc.y,0.0, 1.0) as usize * size as usize / 2)) as usize,
                     (z as usize + (clamp(loc.z,0.0, 1.0) as usize * size as usize / 2)) as usize,
                );
            }
        }

        do_child(
            &self,
            0,
            rootid,
            tree,
            &mut levels,
            self.size,
            0,0,0
        );

        

        tree
    }
}
