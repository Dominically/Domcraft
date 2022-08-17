use std::{collections::HashMap};
use cgmath::Vector3;
use winit::event::VirtualKeyCode;

//TODO swap hashmaps with EnumMap(s).
//TODO make bindings more efficient so I don't have to use searches.
pub struct Controller {
  keys: HashMap<VirtualKeyCode, bool>,
  bindings: HashMap<Control, Vec<VirtualKeyCode>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Control {
  Forward,
  Backward,
  Left,
  Right,
  Up,
  Down
}

// impl Control {
//   fn to_direction_vector(&self) -> Vector3<f32> {
//     match self {
//         Control::Forward => [0.0, 0.0, 1.0],
//         Control::Backward => [0.0, 0.0, -1.0],
//         Control::Left => [-1.0, 0.0, 0.0],
//         Control::Right => [1.0, 0.0, 0.0],
//         Control::Up => [0.0, 1.0, 0.0],
//         Control::Down => [0.0, -1.0, 0.0],
//     }.into()
//   }
// }

impl Controller {
  pub fn new() -> Self {
    Self {
      keys: HashMap::new(),
      bindings: HashMap::new(),
    }
  }

  pub fn set_key(&mut self, key: VirtualKeyCode, state: bool) {
    self.keys.insert(key, state);
  }

  pub fn set_bindings(&mut self, binds: &[(VirtualKeyCode, Control)]){
    for (key, action) in binds {
      match self.bindings.get_mut(action) {
        Some(action_list) => action_list.push(*key),
        None => {
          self.bindings.insert(*action, vec![*key]);
        }
      }
    }
  }

  pub fn get_action(&self, action: Control) -> bool {
    match self.bindings.get(&action) {
        Some(binds) => binds.iter().any(|keycode| {
          match self.keys.get(keycode) {
            Some(state) => *state,
            None => false,
        }
        }),
        None => false,
    }
  }

  pub fn get_action_value<T>(&self, a: (Control, T), b: (Control, T), default: T) -> T {
    let a_pressed = self.get_action(a.0);
    let b_pressed = self.get_action(b.0);

    if a_pressed && !b_pressed {
      return a.1;
    } else if b_pressed && !a_pressed {
      return b.1;
    } else {
      return default;
    }
  }
}