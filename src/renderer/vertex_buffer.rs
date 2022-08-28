use std::{marker::PhantomData, mem::{size_of, replace}};

use bytemuck::{Pod, cast_slice};
use wgpu::{Buffer, Device, Queue, BufferDescriptor, BufferUsages};
///Used for storing data about vertices.
pub struct VertexBuffer<T: Pod> {
  buffer: Buffer,
  buffer_capacity: usize,
  buffer_length: usize,
  buffer_spare_size: usize, //The amount of space to leave spare when creating or resizing the buffer.
  phantom_type: PhantomData<T>
}

impl<T: Pod> VertexBuffer<T> {

  pub fn new(device: &Device, queue: &Queue, contents: &[T]) -> Self {
    const DEFAULT_EXCESS_ITEMS: usize = 400;
    let buffer_spare_size = DEFAULT_EXCESS_ITEMS;
    let buffer_length = contents.len();
    let buffer_capacity = buffer_length + buffer_spare_size;
    

    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (size_of::<T>() * buffer_capacity) as u64,
        usage: BufferUsages::VERTEX,
        mapped_at_creation: false,
    });

    //Send contents to GPU.
    queue.write_buffer(&buffer, 0, cast_slice(contents));

    Self {
        buffer,
        buffer_capacity,
        buffer_length,
        buffer_spare_size,
        phantom_type: PhantomData,
    }
  }

  pub fn update(&mut self, device: &Device, queue: &Queue, contents: &[T]) {
    if contents.len() <= self.buffer_capacity { //Use existing buffer.
      self.buffer_length = contents.len();
      queue.write_buffer(&self.buffer, 0, cast_slice(contents));
    } else {
      let new = Self::new(device, queue, contents);
      replace(self, new);
    }
  } 
}

impl <T: Pod> Drop for VertexBuffer<T> {
  fn drop(&mut self) {
    self.buffer.destroy();
  }
}