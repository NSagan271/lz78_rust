use crate::{
    sequence::Sequence,
    spa::{SPAParams, SPA},
};
use anyhow::Result;
use itertools::Itertools;

pub struct Classifier<S> {
    spas: Vec<S>,
}

impl<S> Classifier<S>
where
    S: SPA,
{
    pub fn train_classifier<T>(&mut self, input: &T, class: u32, params: &SPAParams) -> Result<f64>
    where
        T: Sequence,
    {
        self.spas[class as usize].train_on_block(input, params)
    }

    pub fn train_classifiers_parallel<T>(
        &mut self,
        inputs: &[T],
        classes: &[u32],
        params: &SPAParams,
    ) -> Result<Vec<f64>> {
        let class_to_seqs = classes
            .iter()
            .map(|&x| x)
            .zip(inputs)
            .into_group_map()
            .into_iter();
        let seqs_per_class = (0..self.spas.len() as u32)
            .map(|i| class_to_seqs[&i])
            .collect_vec();

        todo!()
    }
}
