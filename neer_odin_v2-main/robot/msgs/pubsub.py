import zmq
from typing import List, Callable
from robot.msgs.zed_msgs import Message


class Publisher:
    def __init__(self, ctx: zmq.Context, port: int, host: str = "*", HWM: int | None = None):
        self.socket: zmq.Socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")
        if HWM is not None:
            self.socket.setsockopt(zmq.SNDHWM, HWM)

    def publish(self, topic: str, data: Message, copy: bool = True):
        payload = data.serialize()
        self.socket.send_string(topic, zmq.SNDMORE)
        self.socket.send(payload, copy=copy)

    def stop(self):
        self.socket.close()


class Subscriber:
    def __init__(
        self,
        ctx: zmq.Context,
        port: int,
        topics: List[str],
        deserializer: List[Callable[[bytes], Message]],
        host: str = "localhost",
        conflate: bool = True,
        no_block: bool = False,
    ):
        self.socket: zmq.Socket = ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        for topic in topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)

        self.deserializer = {topic: deserializer[i] for i, topic in enumerate(topics)}
        self.no_block = zmq.NOBLOCK if no_block else 0

    # def receive(self) -> tuple[str | None, Message | None]:
    #     try:
    #         payload = self.socket.recv_multipart(flags=self.no_block)
    #         topic = payload[0].decode("utf-8")
    #         data = self.deserializer[topic](payload[1])
    #         return topic, data
    #     except zmq.error.Again:
    #         return None, None

    def receive(self) -> tuple[str | None, Message | None]:
        try:
            payload = self.socket.recv_multipart(flags=self.no_block)

            # Handle pyzmq returning zmq.Frame or bytes
            frame0 = payload[0]
            frame1 = payload[1]

            if hasattr(frame0, "bytes"):
                topic_bytes = frame0.bytes
            else:
                topic_bytes = frame0

            if hasattr(frame1, "bytes"):
                data_bytes = frame1.bytes
            else:
                data_bytes = frame1

            topic = topic_bytes.decode("utf-8")
            data = self.deserializer[topic](data_bytes)
            return topic, data

        except zmq.error.Again:
            return None, None


    def register_poller(self, poller: zmq.Poller):
        poller.register(self.socket, zmq.POLLIN)

    def stop(self):
        self.socket.close()