import pretty_midi


RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

START_IDX = {
    "note_on": 0,
    "note_off": RANGE_NOTE_ON,
    "time_shift": RANGE_NOTE_ON + RANGE_NOTE_OFF,
    "velocity": RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
}


class SustainAdapter:
    def __init__(self, time: float, type: str):
        self.start = time  # 起始时间
        self.type = type  # 类型


class SustainDownManager:
    def __init__(self, start: float, end: float):
        self.start = start  # 持续起始时间
        self.end = end  # 持续结束时间
        self.managed_notes = []  # 被管理的音符列表
        self._note_dict = {}  # 字典：键是音高，值是音符开始时间

    def add_managed_note(self, note: pretty_midi.Note):
        """将音符添加到管理列表"""
        self.managed_notes.append(note)

    def transposition_notes(self):
        """处理延音音符的结束时间，进行必要的转置"""
        for note in reversed(self.managed_notes):
            # 尝试查找相同音高的音符结束时间
            note.end = self._note_dict.get(note.pitch, max(self.end, note.end))
            self._note_dict[note.pitch] = note.start


class SplitNote:
    def __init__(self, type: str, time: float, value: int, velocity: int):
        """
        拆分音符
        :param type: 音符类型 ('note_on' 或 'note_off')
        :param time: 音符发生时间
        :param value: 音符的音高（值）
        :param velocity: 音符的力度
        """
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return f"<[SNote] time: {self.time} type: {self.type}, value: {self.value}, velocity: {self.velocity}>"


class Event:
    def __init__(self, event_type: str, value: int):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return f"<Event type: {self.type}, value: {self.value}>"

    def to_int(self) -> int:
        """将事件转换为整数"""
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value: int) -> "Event":
        """根据整数值恢复事件类型和音符值"""
        info = Event._type_check(int_value)
        return Event(info["type"], info["value"])

    @staticmethod
    def _type_check(int_value: int) -> dict:
        """根据整数值判断事件类型"""
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON + RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON + RANGE_NOTE_OFF, RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)

        if int_value in range_note_on:
            return {"type": "note_on", "value": int_value}
        elif int_value in range_note_off:
            return {"type": "note_off", "value": int_value - RANGE_NOTE_ON}
        elif int_value in range_time_shift:
            return {"type": "time_shift", "value": int_value - RANGE_NOTE_ON - RANGE_NOTE_OFF}
        else:
            return {"type": "velocity", "value": int_value - RANGE_NOTE_ON - RANGE_NOTE_OFF - RANGE_TIME_SHIFT}


def divide_note(notes):
    """
    将音符拆分为 'note_on' 和 'note_off' 事件

    参数:
        notes: 一个按时间顺序排列的 `pretty_midi.Note` 对象列表

    返回:
        返回一个包含 `SplitNote` 对象的列表，每个音符会拆分为 `note_on` 和 `note_off` 两个事件
    """
    notes.sort(key=lambda x: x.start)  # 如果确保传入的 notes 已经排好序，可去掉此行

    result_array = []
    for note in notes:
        # 创建 'note_on' 和 'note_off' 事件
        result_array.extend(
            [
                SplitNote("note_on", note.start, note.pitch, note.velocity),
                SplitNote("note_off", note.end, note.pitch, None),
            ]
        )
    return result_array


def merge_note(snote_sequence):
    """
    将拆分的音符序列合并为原始音符（包括 'note_on' 和 'note_off' 事件）

    参数:
        snote_sequence: 一个包含 'note_on' 和 'note_off' 事件的 `SplitNote` 对象列表
            这些事件应该按时间顺序排列，未考虑音力度。

    返回:
        返回一个 `pretty_midi.Note` 对象的列表，其中包含所有有效的音符。
        每个音符对象代表一个音符的开始和结束时间，以及音高和力度。
    """
    note_on_dict = {}  # 用于存储 'note_on' 事件，按音高（pitch）索引
    result_array = []  # 存储合并后的音符

    for snote in snote_sequence:
        if snote.type == "note_on":
            # 如果是 'note_on'，则将其记录在字典中
            note_on_dict[snote.value] = snote
        elif snote.type == "note_off":
            # 如果是 'note_off'，尝试从字典中查找对应的 'note_on'
            try:
                on = note_on_dict[snote.value]
                off = snote
                # 如果 'note_on' 和 'note_off' 的时间一样，跳过此音符
                if off.time - on.time == 0:
                    continue
                # 创建一个 `pretty_midi.Note` 对象
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except KeyError:
                # 如果无法找到对应的 'note_on'，则输出丢失音符的提示信息
                print(f"丢失音符信息，音高为: {snote.value}")

    return result_array


def snote_to_events(snote: SplitNote, prev_vel: int):
    """
    将一个拆分音符（SplitNote）转换为事件序列（包括可能的速度变化和音符类型事件）。

    参数:
        snote: 一个 `SplitNote` 对象，表示一个拆分音符（包含音符的类型、时间、音高和力度）
        prev_vel: 上一个音符的力度，用于比较和决定是否添加新的速度变化事件

    返回:
        返回一个 `Event` 对象的列表，表示与此音符相关的事件（包括速度变化和音符的 'note_on' 或 'note_off'）。
    """
    result = []

    # 如果当前音符有力度值
    if snote.velocity is not None:
        # 修改力度，通常力度值范围为 0-127，所以我们用 4 来进行缩放
        modified_velocity = snote.velocity // 4
        # 如果当前速度和上一个音符的速度不同，则添加速度变化事件
        if prev_vel != modified_velocity:
            result.append(Event(event_type="velocity", value=modified_velocity))

    # 添加当前音符的类型事件（'note_on' 或 'note_off'）
    result.append(Event(event_type=snote.type, value=snote.value))

    return result


def eventSeq_to_snoteSeq(event_sequence):
    """
    将事件序列转换为拆分音符序列。每个事件代表一个时间偏移、速度变化或音符事件。

    参数:
        event_sequence: 一个包含 `Event` 对象的列表，每个事件表示时间偏移、速度变化或音符的类型。

    返回:
        返回一个 `SplitNote` 对象的列表，代表按时间顺序排列的拆分音符。
    """
    timeline = 0  # 当前时间线（以秒为单位）
    velocity = 0  # 当前力度（单位为 4 的倍数）
    snote_seq = []  # 用于存储拆分音符的列表

    for event in event_sequence:
        if event.type == "time_shift":
            # 更新时间线，time_shift 表示时间的偏移量
            timeline += (event.value + 1) / 100
        elif event.type == "velocity":
            # 更新速度，velocity 代表音符的力度
            velocity = event.value * 4
        else:
            # 创建新的拆分音符（包括 'note_on' 或 'note_off'）
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)

    return snote_seq


def make_time_sift_events(prev_time: float, post_time: float):
    """
    根据给定的前后时间，生成时间偏移事件（time_shift）。每个事件表示时间上的一个小间隔。

    参数:
        prev_time: 前一个时间点（浮动时间），单位为秒
        post_time: 后一个时间点（浮动时间），单位为秒

    返回:
        返回一个 `Event` 对象的列表，表示按时间间隔生成的多个 `time_shift` 事件。
    """
    time_interval = int(round((post_time - prev_time) * 100))  # 将时间差转换为整数（以 1/100 秒为单位）
    results = []

    # 生成完整的 time_shift 事件直到剩余的时间小于 RANGE_TIME_SHIFT
    full_shifts = time_interval // RANGE_TIME_SHIFT
    results.extend([Event(event_type="time_shift", value=RANGE_TIME_SHIFT - 1)] * full_shifts)

    # 处理剩余的时间间隔
    remaining_time = time_interval % RANGE_TIME_SHIFT
    if remaining_time > 0:
        results.append(Event(event_type="time_shift", value=remaining_time - 1))

    return results


def control_preprocess(ctrl_changes, manager=None):
    sustains = []

    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # 踩下延音踏板
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # 松开延音踏板
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            # 再次确认延音踏板结束时间
            sustains[-1].end = ctrl.time
    return sustains


def note_preprocess(susteins, notes):
    note_stream = []

    if susteins:  # 如果 MIDI 文件包含延音控制
        for sustain in susteins:
            for note_idx, note in enumerate(notes):
                if note.start < sustain.start:
                    note_stream.append(note)
                elif note.start > sustain.end:
                    notes = notes[note_idx:]
                    sustain.transposition_notes()
                    break
                else:
                    sustain.add_managed_note(note)

        for sustain in susteins:
            note_stream += sustain.managed_notes

    else:  # 如果没有延音控制，直接加入音符流
        for note_idx, note in enumerate(notes):
            note_stream.append(note)

    note_stream.sort(key=lambda x: x.start)
    return note_stream


def encode_midi(file_path):
    """将 MIDI 文件编码为整数事件序列"""
    midi = pretty_midi.PrettyMIDI(midi_file=file_path)
    all_notes = []

    for instrument in midi.instruments:
        # 仅处理延音控制器（Control Number 64）
        sustain_ctrls = control_preprocess([ctrl for ctrl in instrument.control_changes if ctrl.number == 64])
        all_notes.extend(note_preprocess(sustain_ctrls, instrument.notes))

    # 拆分 note_on 和 note_off 事件
    split_notes = divide_note(all_notes)
    split_notes.sort(key=lambda note: note.time)

    events = []
    current_time = 0
    current_velocity = 0

    for snote in split_notes:
        # 插入时间偏移事件
        events.extend(make_time_sift_events(prev_time=current_time, post_time=snote.time))
        # 插入音符事件和速度变更事件
        events.extend(snote_to_events(snote=snote, prev_vel=current_velocity))

        current_time = snote.time
        current_velocity = snote.velocity

    return [event.to_int() for event in events]


def decode_midi(idx_array, file_path=None) -> pretty_midi.PrettyMIDI:
    """
    将事件索引数组解码为 MIDI 文件。

    参数:
        idx_array: 事件索引数组，每个索引值代表一个 MIDI 事件。
        file_path: 如果提供文件路径，将会把生成的 MIDI 文件保存到该路径。默认不保存。

    返回:
        返回一个 `pretty_midi.PrettyMIDI` 对象，表示解码后的 MIDI 数据。
    """
    # 从事件索引数组生成事件序列
    event_sequence = [Event.from_int(idx) for idx in idx_array]

    # 将事件序列转换为拆分音符序列
    snote_seq = eventSeq_to_snoteSeq(event_sequence)

    # 合并音符并排序
    note_seq = merge_note(snote_seq)
    note_seq.sort(key=lambda x: x.start)  # 确保音符按开始时间排序

    # 创建 PrettyMIDI 对象，并将音符加入其中
    mid = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=1, is_drum=False, name="Bright Acoustic Piano")
    instrument.notes = note_seq

    # 将乐器加入到 MIDI 中
    mid.instruments.append(instrument)

    # 如果指定了保存路径，写入文件
    if file_path:
        mid.write(file_path)

    return mid



if __name__ == "__main__":
    """
    仅供文件导入测试
    """
    encoded = encode_midi("MidiFiles/Melody.mid")
    print(encoded)
    decided = decode_midi(encoded, file_path="MidiFiles/processor_test.mid")

    ins = pretty_midi.PrettyMIDI("MidiFiles/Melody.mid")
    print(ins)
    print(ins.instruments[0])
    for i in ins.instruments:
        print(i.control_changes)
        print(i.notes)
