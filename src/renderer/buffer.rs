use std::{marker::PhantomData, mem::size_of, sync::Arc};

use bytemuck::{Pod, cast_slice};
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Device, Label, Queue, ShaderStages};

use super::Descriptable;

//General buffer for either vertices or indices. (or anything else I happen to need it for).
pub struct ArrayBuffer<T: Pod> {
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
  // Uniform
}

impl<T: Pod> ArrayBuffer<T> {
  ///Creates a new buffer.
  pub fn new(device: &Device, queue: &Queue, typ: GenericBufferType, contents: &[T], spare_reserve: usize) -> Self {
    let length = contents.len();
    let size = length + spare_reserve;

    let usage = match typ {
        GenericBufferType::Vertex => BufferUsages::VERTEX,
        GenericBufferType::Index => BufferUsages::INDEX,
        // GenericBufferType::Uniform => BufferUsages::UNIFORM,
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

pub struct UniformBuffer<T: Pod> {
  buf: Buffer,
  bind_group_layout: BindGroupLayout,
  bind_group: BindGroup,
  typ: PhantomData<T>
}

pub enum UniformBufferUsage {
  Vertex,
  Fragment
}

impl<T: Pod> UniformBuffer<T> {
  pub fn new(device: &Device, typ: UniformBufferUsage, label: Option<&str>) -> Self{
    let buf = device.create_buffer(&BufferDescriptor {
        label: label.map(|s| format!("{s} buffer.")).as_deref(),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    //TODO split buffers and bind groups apart.

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: label.map(|s| format!("{s} bind group layout.")).as_deref(),
        entries: &[BindGroupLayoutEntry {
          binding: 0,//At the moment only one buffer per binding. I will increase it when I can be bothered.
          visibility: match typ {
            UniformBufferUsage::Vertex => ShaderStages::VERTEX,
            UniformBufferUsage::Fragment => ShaderStages::FRAGMENT,
          },
          ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None 
          },
          count: None, 
        }],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: label.map(|s| format!("{s} bind group.")).as_deref(),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
          binding: 0,
          resource: buf.as_entire_binding()
        }],
    });

    Self {
      buf,
      bind_group_layout,
      bind_group,
      typ: PhantomData
    }
  }

  pub fn get_buffer(&self) -> &Buffer {
    &self.buf
  }

  pub fn get_bind_group_layout(&self) -> &BindGroupLayout {
    &self.bind_group_layout
  }

  pub fn get_bind_group(&self) -> &BindGroup {
    &self.bind_group
  }

  pub fn update(&self, queue: &Queue, data: T) {
    queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(&[data]));
  }
}