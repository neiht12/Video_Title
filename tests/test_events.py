from __future__ import annotations

import unittest
from fractions import Fraction

import cv2
import numpy as np

from src.events import build_events, EventConfig, fingerprint_distance, line_signature, track_observations
from src.schemas import FrameObservation, LineDetection


def observation(
    index: int,
    boxes: tuple[tuple[float, float, float, float], ...] = (),
    texts: tuple[str, ...] = (),
    styles: tuple[float, ...] = (),
) -> FrameObservation:
    return FrameObservation(
        frame_index=index, source_pts=index * 512, time_sec=Fraction(index, 30),
        lines=tuple(LineDetection(box, 0.95) for box in boxes),
        fingerprints=texts, style_scores=styles,
    )


MAIN = (100, 840, 600, 900)
TIAN = (325, 900, 395, 950)
SIZE = (720, 1280)


class EventTests(unittest.TestCase):
    def test_build_events_exposes_shared_frame_timeline(self) -> None:
        frames = [observation(i, (MAIN,), ("same",)) for i in range(3)]
        frames.append(observation(3))
        events, timeline = build_events(frames, SIZE)
        self.assertEqual(len(events), 1)
        self.assertEqual([item.frame_index for item in timeline], list(range(4)))
        self.assertEqual([item.source_pts for item in timeline], [0, 512, 1024, 1536])
        self.assertEqual(timeline[-1].boxes_xyxy, ())

    def test_text_style_detects_bright_strokes_with_dark_outline(self) -> None:
        image = np.full((100, 200, 3), (85, 130, 170), dtype=np.uint8)
        box = (20, 20, 180, 80)
        plain_score, _ = line_signature(image, box)
        cv2.putText(image, "TIAN", (27, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(image, "TIAN", (27, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 255, 255), 3, cv2.LINE_AA)
        text_score, _ = line_signature(image, box)
        self.assertLess(plain_score, 0.01)
        self.assertGreater(text_score, 0.04)

    def test_short_line_fingerprint_ignores_detection_box_jitter(self) -> None:
        image = np.full((128, 128, 3), 35, dtype=np.uint8)
        cv2.rectangle(image, (49, 49), (78, 78), (255, 255, 255), 4)
        cv2.line(image, (64, 51), (64, 76), (255, 255, 255), 3)
        cv2.line(image, (51, 64), (76, 64), (255, 255, 255), 3)
        signatures = [line_signature(image, box)[1] for box in (
            (45, 45, 83, 83), (44, 46, 82, 84), (46, 44, 84, 82),
        )]
        self.assertTrue(all(fingerprint_distance(signatures[0], value) < 0.1
                            for value in signatures[1:]))

    def test_jitter_is_smoothed_within_one_event(self) -> None:
        shifts = [0, 3, -2, 2, -1, 1]
        frames = [observation(i, ((100 + dx, 840 + dx, 600 + dx, 900 + dx),), ("same",))
                  for i, dx in enumerate(shifts)]
        result = track_observations(frames, SIZE)
        self.assertEqual(len(result.events), 1)
        self.assertEqual((result.events[0].start_frame, result.events[0].end_frame_exclusive), (0, 6))
        raw_left = [frame.raw_boxes[0][0] for frame in result.frames]
        drawn_left = [frame.draw_boxes[0][0] for frame in result.frames]
        self.assertLess(max(drawn_left) - min(drawn_left), max(raw_left) - min(raw_left))
        for frame in result.frames:
            self.assertLessEqual(frame.draw_boxes[0][0], frame.raw_boxes[0][0])
            self.assertGreaterEqual(frame.draw_boxes[0][2], frame.raw_boxes[0][2])

    def test_two_lines_and_narrow_tian_have_independent_events(self) -> None:
        frames = [observation(i, (MAIN, TIAN), ("main", "tian"), (0.18, 0.12))
                  for i in range(5)]
        result = track_observations(frames, SIZE)
        self.assertEqual(len(result.events), 2)
        self.assertTrue(all(len(frame.draw_boxes) == 2 for frame in result.frames))
        self.assertTrue(all(frame.draw_boxes[1][2] - frame.draw_boxes[1][0] < 100
                            for frame in result.frames))
        self.assertNotEqual(result.frames[0].event_ids[0], result.frames[0].event_ids[1])

    def test_one_and_two_frame_gaps_fill_only_with_both_anchors(self) -> None:
        for gap in (1, 2):
            with self.subTest(gap=gap):
                frames = [observation(i, (MAIN,), ("same",)) for i in range(2)]
                frames += [observation(i) for i in range(2, 2 + gap)]
                frames += [observation(i, (MAIN,), ("same",)) for i in range(2 + gap, 5 + gap)]
                frames += [observation(5 + gap)]
                result = track_observations(frames, SIZE)
                self.assertEqual(len(result.events), 1)
                self.assertEqual(result.events[0].filled_frames, gap)
                for index in range(2, 2 + gap):
                    self.assertEqual(len(result.frames[index].raw_boxes), 0)
                    self.assertEqual(len(result.frames[index].draw_boxes), 1)
                self.assertEqual(result.frames[-1].draw_boxes, ())

    def test_long_gap_creates_new_event_without_interpolation(self) -> None:
        frames = [observation(i, (MAIN,), ("same",)) for i in range(3)]
        frames += [observation(i) for i in range(3, 6)]
        frames += [observation(i, (MAIN,), ("same",)) for i in range(6, 9)]
        result = track_observations(frames, SIZE)
        self.assertEqual(len(result.events), 2)
        self.assertEqual([(event.start_frame, event.end_frame_exclusive)
                          for event in result.events], [(0, 3), (6, 9)])
        self.assertTrue(all(not result.frames[index].draw_boxes for index in (3, 4, 5)))

    def test_confirmed_new_sentence_at_same_position_splits_event(self) -> None:
        frames = [observation(i, (MAIN,), ("old",)) for i in range(4)]
        frames += [observation(i, (MAIN,), ("new",)) for i in range(4, 8)]
        result = track_observations(frames, SIZE)
        self.assertEqual([(event.start_frame, event.end_frame_exclusive)
                          for event in result.events], [(0, 4), (4, 8)])
        self.assertNotEqual(result.frames[3].event_ids, result.frames[4].event_ids)

    def test_gap_at_sentence_change_is_not_filled_across_events(self) -> None:
        frames = [observation(i, (MAIN,), ("old",)) for i in range(3)]
        frames.append(observation(3))
        frames += [observation(i, (MAIN,), ("new",)) for i in range(4, 8)]
        result = track_observations(frames, SIZE)
        self.assertEqual([(event.start_frame, event.end_frame_exclusive)
                          for event in result.events], [(0, 3), (4, 8)])
        self.assertEqual(result.frames[3].draw_boxes, ())

    def test_single_fingerprint_glitch_does_not_split(self) -> None:
        texts = ["same", "same", "same", "glitch", "same", "same", "same"]
        result = track_observations(
            [observation(i, (MAIN,), (text,)) for i, text in enumerate(texts)], SIZE)
        self.assertEqual(len(result.events), 1)

    def test_short_transient_does_not_draw_and_no_trailing_hold(self) -> None:
        frames = [observation(i, (MAIN,), ("noise",)) if i < 2 else observation(i)
                  for i in range(6)]
        result = track_observations(frames, SIZE)
        self.assertEqual(result.events, ())
        self.assertTrue(all(not frame.draw_boxes for frame in result.frames))
        self.assertIn((0, "transient_track"), result.frames[0].rejected)

    def test_style_and_band_reject_known_false_box_shapes_but_keep_tian(self) -> None:
        button = (419, 971, 441, 988)
        hand = (54, 745, 145, 869)
        foreground = (0, 702, 150, 871)
        frames = [observation(i, (TIAN, button, hand, foreground),
                              ("tian", "button", "hand", "figure"),
                              (0.12, 0.0, 0.001, 0.0)) for i in range(4)]
        result = track_observations(frames, SIZE)
        self.assertEqual(len(result.events), 1)
        self.assertTrue(all(len(frame.draw_boxes) == 1 for frame in result.frames))
        self.assertEqual(set(result.frames[0].rejected), {
            (1, "weak_text_style"), (2, "outside_subtitle_band"),
            (3, "outside_subtitle_band"),
        })

    def test_stationary_textlike_sign_remains_an_explicit_limitation(self) -> None:
        sign = (110, 840, 600, 900)
        frames = [observation(i, (sign,), ("sign",), (0.20,)) for i in range(5)]
        result = track_observations(frames, SIZE)
        self.assertEqual(len(result.events), 1)
        self.assertTrue(all(frame.draw_boxes for frame in result.frames))


if __name__ == "__main__":
    unittest.main()
