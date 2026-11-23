<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 })
const paths: Record<string, string[]> = {
  search: ['m21 21-4.35-4.35', 'M19 10.5a8.5 8.5 0 1 1-17 0 8.5 8.5 0 0 1 17 0'],
  layers: ['m12 3 9 5-9 5-9-5 9-5Z', 'm3 12 9 5 9-5', 'm3 16 9 5 9-5'],
  plus: ['M12 5v14', 'M5 12h14'],
  history: ['M3 11a9 9 0 1 1 2.7 7', 'M3 4v7h7', 'M12 7v5l3 2'],
  book: ['M4 3h13a3 3 0 0 1 3 3v15H6a3 3 0 0 1-3-3V4a1 1 0 0 1 1-1Z', 'M3 17a3 3 0 0 1 3-3h14', 'M8 7h7'],
  help: ['M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4', 'M12 17h.01', 'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0'],
  chevron: ['m9 5 7 7-7 7'],
  down: ['m6 9 6 6 6-6'],
  check: ['m5 12 4 4L19 6'],
  shield: ['m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z', 'm8 12 3 3 5-6'],
  file: ['M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z', 'M14 2v6h6', 'M8 13h8', 'M8 17h5'],
  network: ['M12 4v5', 'M5 15v-3h14v3', 'M3 16h4v4H3z', 'M17 16h4v4h-4z', 'M10 2h4v4h-4z', 'M10 16h4v4h-4z', 'M12 12v3'],
  clock: ['M12 8v4l3 2', 'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0'],
  spark: ['m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z'],
  copy: ['M9 9h11v12H9z', 'M5 15H3V3h12v2'],
  download: ['M12 3v12', 'm7 10 5 5 5-5', 'M5 17v4h14v-4'],
  close: ['m6 6 12 12', 'M6 18 18 6'],
  refresh: ['M20 7a9 9 0 1 0 1 9', 'M21 3v6h-6'],
  menu: ['M4 6h16', 'M4 12h16', 'M4 18h16'],
  external: ['M14 3h7v7', 'm10 14 11-11', 'M10 3H3v18h18v-7'],
  trash: ['M3 6h18', 'M9 6V3h6v3', 'm5 6 1 15h12l1-15', 'M10 10v7', 'M14 10v7'],
  info: ['M12 11v6', 'M12 7h.01', 'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0'],
}
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" class="app-icon">
    <path v-for="(path, index) in paths[name] || paths.file" :key="index" :d="path" />
  </svg>
</template>
