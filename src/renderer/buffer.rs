use std::{marker::PhantomData, mem::size_of, sync::Arc};

use bytemuck::{Pod, cast_slice};
use wgpu::{Buffer, Device, Queue, BufferDescriptor, BufferUsages};

//General buffer for either vertices or indices. (or anything else I happen to need it for).
pub struct GenericBuffer<T: Pod> {
  buf: Arc<Buffer>,
  size: usize,
  length: usize,
  spare: usize,
  typ: GenericBufferType,
  phantom_type: PhantomData<T>
}

#[derive(Clone, Copy)]
pub enum GenericBufferType {
  Vertex,
  Index,
  Uniform
}

impl<T: Pod> GenericBuffer<T> {
  ///Creates a new buffer.
  pub fn new(device: &Device, queue: &Queue, typ: GenericBufferType, contents: &[T], spare_reserve: usize) -> Self {
    let length = contents.len();
    let size = length + spare_reserve;

    let usage = match typ {
        GenericBufferType::Vertex => BufferUsages::VERTEX,
        GenericBufferType::Index => BufferUsages::INDEX,
        GenericBufferType::Uniform => BufferUsages::UNIFORM,
        
    };

    let buf = Arc::new(device.create_buffer(&BufferDescriptor {
        label: None,
        size: (size * size_of::<T>()) as u64,
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    queue.write_buffer(&buf, 0, cast_slice(contents));

    Self {
        buf,
        size,
        length,
        spare: spare_reserve,
        typ,
        phantom_type: PhantomData,
    }
  }

  ///Updates the buffer and reallocates if it cannot fit in the existing space.
  pub fn update(&mut self, device: &Device, queue: &Queue, contents: &[T]) {
    if contents.len() <= self.size { //Reuse existing buffer.
      self.length = contents.len();
      queue.write_buffer(&self.buf, 0, cast_slice(contents));
    } else { //Create new buffer
      *self = Self::new(device, queue, self.typ, contents, self.spare);
    }
  }

  pub fn get_buffer(&self) -> Arc<Buffer> {
    self.buf.clone()
  }

  pub fn len(&self) -> usize {
    self.length
  }
}

// impl<T: Pod> Drop for GenericBuffer<T> {
//     fn drop(&mut self) {
//         self.buf.destroy();
//     }
// }