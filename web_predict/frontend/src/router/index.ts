import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'predict',
      component: () => import('../views/PredictView.vue'),
      meta: { title: '性质预测', keepAlive: true },
    },
    {
      path: '/train',
      name: 'train',
      component: () => import('../views/TrainView.vue'),
      meta: { title: '模型训练', keepAlive: true },
    },
  ],
})

export default router
