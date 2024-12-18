// source: state.proto
/**
 * @fileoverview
 * @enhanceable
 * @suppress {missingRequire} reports error on implicit type usages.
 * @suppress {messageConventions} JS Compiler reports an error if a variable or
 *     field starts with 'MSG_' and isn't a translatable message.
 * @public
 */
// GENERATED CODE -- DO NOT EDIT!
/* eslint-disable */
// @ts-nocheck

var jspb = require('google-protobuf');
var goog = jspb;
var global = (function() { return this || window || global || self || Function('return this')(); }).call(null);

var history_pb = require('./history_pb.js');
goog.object.extend(proto, history_pb);
var enums_pb = require('./enums_pb.js');
goog.object.extend(proto, enums_pb);
goog.exportSymbol('proto.rlenv.Info', null, global);
goog.exportSymbol('proto.rlenv.State', null, global);
/**
 * Generated by JsPbCodeGenerator.
 * @param {Array=} opt_data Optional initial data array, typically from a
 * server response, or constructed directly in Javascript. The array is used
 * in place and becomes part of the constructed object. It is not cloned.
 * If no data is provided, the constructed object will be empty, but still
 * valid.
 * @extends {jspb.Message}
 * @constructor
 */
proto.rlenv.Info = function(opt_data) {
  jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.rlenv.Info, jspb.Message);
if (goog.DEBUG && !COMPILED) {
  /**
   * @public
   * @override
   */
  proto.rlenv.Info.displayName = 'proto.rlenv.Info';
}
/**
 * Generated by JsPbCodeGenerator.
 * @param {Array=} opt_data Optional initial data array, typically from a
 * server response, or constructed directly in Javascript. The array is used
 * in place and becomes part of the constructed object. It is not cloned.
 * If no data is provided, the constructed object will be empty, but still
 * valid.
 * @extends {jspb.Message}
 * @constructor
 */
proto.rlenv.State = function(opt_data) {
  jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.rlenv.State, jspb.Message);
if (goog.DEBUG && !COMPILED) {
  /**
   * @public
   * @override
   */
  proto.rlenv.State.displayName = 'proto.rlenv.State';
}



if (jspb.Message.GENERATE_TO_OBJECT) {
/**
 * Creates an object representation of this proto.
 * Field names that are reserved in JavaScript and will be renamed to pb_name.
 * Optional fields that are not set will be set to undefined.
 * To access a reserved field use, foo.pb_<name>, eg, foo.pb_default.
 * For the list of reserved names please see:
 *     net/proto2/compiler/js/internal/generator.cc#kKeyword.
 * @param {boolean=} opt_includeInstance Deprecated. whether to include the
 *     JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @return {!Object}
 */
proto.rlenv.Info.prototype.toObject = function(opt_includeInstance) {
  return proto.rlenv.Info.toObject(opt_includeInstance, this);
};


/**
 * Static version of the {@see toObject} method.
 * @param {boolean|undefined} includeInstance Deprecated. Whether to include
 *     the JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @param {!proto.rlenv.Info} msg The msg instance to transform.
 * @return {!Object}
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.rlenv.Info.toObject = function(includeInstance, msg) {
  var f, obj = {
    gameid: jspb.Message.getFieldWithDefault(msg, 1, 0),
    done: jspb.Message.getBooleanFieldWithDefault(msg, 2, false),
    winreward: jspb.Message.getFloatingPointFieldWithDefault(msg, 3, 0.0),
    hpreward: jspb.Message.getFloatingPointFieldWithDefault(msg, 4, 0.0),
    playerindex: jspb.Message.getBooleanFieldWithDefault(msg, 5, false),
    turn: jspb.Message.getFieldWithDefault(msg, 6, 0),
    turnssinceswitch: jspb.Message.getFieldWithDefault(msg, 7, 0),
    heuristicaction: jspb.Message.getFieldWithDefault(msg, 8, 0),
    lastaction: jspb.Message.getFieldWithDefault(msg, 9, 0),
    lastmove: jspb.Message.getFieldWithDefault(msg, 10, 0),
    faintedreward: jspb.Message.getFloatingPointFieldWithDefault(msg, 11, 0.0),
    heuristicdist: msg.getHeuristicdist_asB64(),
    switchreward: jspb.Message.getFloatingPointFieldWithDefault(msg, 13, 0.0),
    ts: jspb.Message.getFloatingPointFieldWithDefault(msg, 14, 0.0),
    longevityreward: jspb.Message.getFloatingPointFieldWithDefault(msg, 15, 0.0),
    drawratio: jspb.Message.getFloatingPointFieldWithDefault(msg, 16, 0.0)
  };

  if (includeInstance) {
    obj.$jspbMessageInstance = msg;
  }
  return obj;
};
}


/**
 * Deserializes binary data (in protobuf wire format).
 * @param {jspb.ByteSource} bytes The bytes to deserialize.
 * @return {!proto.rlenv.Info}
 */
proto.rlenv.Info.deserializeBinary = function(bytes) {
  var reader = new jspb.BinaryReader(bytes);
  var msg = new proto.rlenv.Info;
  return proto.rlenv.Info.deserializeBinaryFromReader(msg, reader);
};


/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.rlenv.Info} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.rlenv.Info}
 */
proto.rlenv.Info.deserializeBinaryFromReader = function(msg, reader) {
  while (reader.nextField()) {
    if (reader.isEndGroup()) {
      break;
    }
    var field = reader.getFieldNumber();
    switch (field) {
    case 1:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setGameid(value);
      break;
    case 2:
      var value = /** @type {boolean} */ (reader.readBool());
      msg.setDone(value);
      break;
    case 3:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setWinreward(value);
      break;
    case 4:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setHpreward(value);
      break;
    case 5:
      var value = /** @type {boolean} */ (reader.readBool());
      msg.setPlayerindex(value);
      break;
    case 6:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setTurn(value);
      break;
    case 7:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setTurnssinceswitch(value);
      break;
    case 8:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setHeuristicaction(value);
      break;
    case 9:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setLastaction(value);
      break;
    case 10:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setLastmove(value);
      break;
    case 11:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setFaintedreward(value);
      break;
    case 12:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setHeuristicdist(value);
      break;
    case 13:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setSwitchreward(value);
      break;
    case 14:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setTs(value);
      break;
    case 15:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setLongevityreward(value);
      break;
    case 16:
      var value = /** @type {number} */ (reader.readFloat());
      msg.setDrawratio(value);
      break;
    default:
      reader.skipField();
      break;
    }
  }
  return msg;
};


/**
 * Serializes the message to binary data (in protobuf wire format).
 * @return {!Uint8Array}
 */
proto.rlenv.Info.prototype.serializeBinary = function() {
  var writer = new jspb.BinaryWriter();
  proto.rlenv.Info.serializeBinaryToWriter(this, writer);
  return writer.getResultBuffer();
};


/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.rlenv.Info} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.rlenv.Info.serializeBinaryToWriter = function(message, writer) {
  var f = undefined;
  f = message.getGameid();
  if (f !== 0) {
    writer.writeInt32(
      1,
      f
    );
  }
  f = message.getDone();
  if (f) {
    writer.writeBool(
      2,
      f
    );
  }
  f = message.getWinreward();
  if (f !== 0.0) {
    writer.writeFloat(
      3,
      f
    );
  }
  f = message.getHpreward();
  if (f !== 0.0) {
    writer.writeFloat(
      4,
      f
    );
  }
  f = message.getPlayerindex();
  if (f) {
    writer.writeBool(
      5,
      f
    );
  }
  f = message.getTurn();
  if (f !== 0) {
    writer.writeInt32(
      6,
      f
    );
  }
  f = message.getTurnssinceswitch();
  if (f !== 0) {
    writer.writeInt32(
      7,
      f
    );
  }
  f = message.getHeuristicaction();
  if (f !== 0) {
    writer.writeInt32(
      8,
      f
    );
  }
  f = message.getLastaction();
  if (f !== 0) {
    writer.writeInt32(
      9,
      f
    );
  }
  f = message.getLastmove();
  if (f !== 0) {
    writer.writeInt32(
      10,
      f
    );
  }
  f = message.getFaintedreward();
  if (f !== 0.0) {
    writer.writeFloat(
      11,
      f
    );
  }
  f = message.getHeuristicdist_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      12,
      f
    );
  }
  f = message.getSwitchreward();
  if (f !== 0.0) {
    writer.writeFloat(
      13,
      f
    );
  }
  f = message.getTs();
  if (f !== 0.0) {
    writer.writeFloat(
      14,
      f
    );
  }
  f = message.getLongevityreward();
  if (f !== 0.0) {
    writer.writeFloat(
      15,
      f
    );
  }
  f = message.getDrawratio();
  if (f !== 0.0) {
    writer.writeFloat(
      16,
      f
    );
  }
};


/**
 * optional int32 gameId = 1;
 * @return {number}
 */
proto.rlenv.Info.prototype.getGameid = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 1, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setGameid = function(value) {
  return jspb.Message.setProto3IntField(this, 1, value);
};


/**
 * optional bool done = 2;
 * @return {boolean}
 */
proto.rlenv.Info.prototype.getDone = function() {
  return /** @type {boolean} */ (jspb.Message.getBooleanFieldWithDefault(this, 2, false));
};


/**
 * @param {boolean} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setDone = function(value) {
  return jspb.Message.setProto3BooleanField(this, 2, value);
};


/**
 * optional float winReward = 3;
 * @return {number}
 */
proto.rlenv.Info.prototype.getWinreward = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 3, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setWinreward = function(value) {
  return jspb.Message.setProto3FloatField(this, 3, value);
};


/**
 * optional float hpReward = 4;
 * @return {number}
 */
proto.rlenv.Info.prototype.getHpreward = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 4, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setHpreward = function(value) {
  return jspb.Message.setProto3FloatField(this, 4, value);
};


/**
 * optional bool playerIndex = 5;
 * @return {boolean}
 */
proto.rlenv.Info.prototype.getPlayerindex = function() {
  return /** @type {boolean} */ (jspb.Message.getBooleanFieldWithDefault(this, 5, false));
};


/**
 * @param {boolean} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setPlayerindex = function(value) {
  return jspb.Message.setProto3BooleanField(this, 5, value);
};


/**
 * optional int32 turn = 6;
 * @return {number}
 */
proto.rlenv.Info.prototype.getTurn = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 6, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setTurn = function(value) {
  return jspb.Message.setProto3IntField(this, 6, value);
};


/**
 * optional int32 turnsSinceSwitch = 7;
 * @return {number}
 */
proto.rlenv.Info.prototype.getTurnssinceswitch = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 7, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setTurnssinceswitch = function(value) {
  return jspb.Message.setProto3IntField(this, 7, value);
};


/**
 * optional int32 heuristicAction = 8;
 * @return {number}
 */
proto.rlenv.Info.prototype.getHeuristicaction = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 8, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setHeuristicaction = function(value) {
  return jspb.Message.setProto3IntField(this, 8, value);
};


/**
 * optional int32 lastAction = 9;
 * @return {number}
 */
proto.rlenv.Info.prototype.getLastaction = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 9, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setLastaction = function(value) {
  return jspb.Message.setProto3IntField(this, 9, value);
};


/**
 * optional int32 lastMove = 10;
 * @return {number}
 */
proto.rlenv.Info.prototype.getLastmove = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 10, 0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setLastmove = function(value) {
  return jspb.Message.setProto3IntField(this, 10, value);
};


/**
 * optional float faintedReward = 11;
 * @return {number}
 */
proto.rlenv.Info.prototype.getFaintedreward = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 11, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setFaintedreward = function(value) {
  return jspb.Message.setProto3FloatField(this, 11, value);
};


/**
 * optional bytes heuristicDist = 12;
 * @return {!(string|Uint8Array)}
 */
proto.rlenv.Info.prototype.getHeuristicdist = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 12, ""));
};


/**
 * optional bytes heuristicDist = 12;
 * This is a type-conversion wrapper around `getHeuristicdist()`
 * @return {string}
 */
proto.rlenv.Info.prototype.getHeuristicdist_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getHeuristicdist()));
};


/**
 * optional bytes heuristicDist = 12;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getHeuristicdist()`
 * @return {!Uint8Array}
 */
proto.rlenv.Info.prototype.getHeuristicdist_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getHeuristicdist()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setHeuristicdist = function(value) {
  return jspb.Message.setProto3BytesField(this, 12, value);
};


/**
 * optional float switchReward = 13;
 * @return {number}
 */
proto.rlenv.Info.prototype.getSwitchreward = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 13, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setSwitchreward = function(value) {
  return jspb.Message.setProto3FloatField(this, 13, value);
};


/**
 * optional float ts = 14;
 * @return {number}
 */
proto.rlenv.Info.prototype.getTs = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 14, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setTs = function(value) {
  return jspb.Message.setProto3FloatField(this, 14, value);
};


/**
 * optional float longevityReward = 15;
 * @return {number}
 */
proto.rlenv.Info.prototype.getLongevityreward = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 15, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setLongevityreward = function(value) {
  return jspb.Message.setProto3FloatField(this, 15, value);
};


/**
 * optional float drawRatio = 16;
 * @return {number}
 */
proto.rlenv.Info.prototype.getDrawratio = function() {
  return /** @type {number} */ (jspb.Message.getFloatingPointFieldWithDefault(this, 16, 0.0));
};


/**
 * @param {number} value
 * @return {!proto.rlenv.Info} returns this
 */
proto.rlenv.Info.prototype.setDrawratio = function(value) {
  return jspb.Message.setProto3FloatField(this, 16, value);
};





if (jspb.Message.GENERATE_TO_OBJECT) {
/**
 * Creates an object representation of this proto.
 * Field names that are reserved in JavaScript and will be renamed to pb_name.
 * Optional fields that are not set will be set to undefined.
 * To access a reserved field use, foo.pb_<name>, eg, foo.pb_default.
 * For the list of reserved names please see:
 *     net/proto2/compiler/js/internal/generator.cc#kKeyword.
 * @param {boolean=} opt_includeInstance Deprecated. whether to include the
 *     JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @return {!Object}
 */
proto.rlenv.State.prototype.toObject = function(opt_includeInstance) {
  return proto.rlenv.State.toObject(opt_includeInstance, this);
};


/**
 * Static version of the {@see toObject} method.
 * @param {boolean|undefined} includeInstance Deprecated. Whether to include
 *     the JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @param {!proto.rlenv.State} msg The msg instance to transform.
 * @return {!Object}
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.rlenv.State.toObject = function(includeInstance, msg) {
  var f, obj = {
    info: (f = msg.getInfo()) && proto.rlenv.Info.toObject(includeInstance, f),
    legalactions: msg.getLegalactions_asB64(),
    history: (f = msg.getHistory()) && history_pb.History.toObject(includeInstance, f),
    moveset: msg.getMoveset_asB64(),
    team: msg.getTeam_asB64(),
    key: jspb.Message.getFieldWithDefault(msg, 6, "")
  };

  if (includeInstance) {
    obj.$jspbMessageInstance = msg;
  }
  return obj;
};
}


/**
 * Deserializes binary data (in protobuf wire format).
 * @param {jspb.ByteSource} bytes The bytes to deserialize.
 * @return {!proto.rlenv.State}
 */
proto.rlenv.State.deserializeBinary = function(bytes) {
  var reader = new jspb.BinaryReader(bytes);
  var msg = new proto.rlenv.State;
  return proto.rlenv.State.deserializeBinaryFromReader(msg, reader);
};


/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.rlenv.State} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.rlenv.State}
 */
proto.rlenv.State.deserializeBinaryFromReader = function(msg, reader) {
  while (reader.nextField()) {
    if (reader.isEndGroup()) {
      break;
    }
    var field = reader.getFieldNumber();
    switch (field) {
    case 1:
      var value = new proto.rlenv.Info;
      reader.readMessage(value,proto.rlenv.Info.deserializeBinaryFromReader);
      msg.setInfo(value);
      break;
    case 2:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setLegalactions(value);
      break;
    case 3:
      var value = new history_pb.History;
      reader.readMessage(value,history_pb.History.deserializeBinaryFromReader);
      msg.setHistory(value);
      break;
    case 4:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setMoveset(value);
      break;
    case 5:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setTeam(value);
      break;
    case 6:
      var value = /** @type {string} */ (reader.readString());
      msg.setKey(value);
      break;
    default:
      reader.skipField();
      break;
    }
  }
  return msg;
};


/**
 * Serializes the message to binary data (in protobuf wire format).
 * @return {!Uint8Array}
 */
proto.rlenv.State.prototype.serializeBinary = function() {
  var writer = new jspb.BinaryWriter();
  proto.rlenv.State.serializeBinaryToWriter(this, writer);
  return writer.getResultBuffer();
};


/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.rlenv.State} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.rlenv.State.serializeBinaryToWriter = function(message, writer) {
  var f = undefined;
  f = message.getInfo();
  if (f != null) {
    writer.writeMessage(
      1,
      f,
      proto.rlenv.Info.serializeBinaryToWriter
    );
  }
  f = message.getLegalactions_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      2,
      f
    );
  }
  f = message.getHistory();
  if (f != null) {
    writer.writeMessage(
      3,
      f,
      history_pb.History.serializeBinaryToWriter
    );
  }
  f = message.getMoveset_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      4,
      f
    );
  }
  f = message.getTeam_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      5,
      f
    );
  }
  f = message.getKey();
  if (f.length > 0) {
    writer.writeString(
      6,
      f
    );
  }
};


/**
 * optional Info info = 1;
 * @return {?proto.rlenv.Info}
 */
proto.rlenv.State.prototype.getInfo = function() {
  return /** @type{?proto.rlenv.Info} */ (
    jspb.Message.getWrapperField(this, proto.rlenv.Info, 1));
};


/**
 * @param {?proto.rlenv.Info|undefined} value
 * @return {!proto.rlenv.State} returns this
*/
proto.rlenv.State.prototype.setInfo = function(value) {
  return jspb.Message.setWrapperField(this, 1, value);
};


/**
 * Clears the message field making it undefined.
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.clearInfo = function() {
  return this.setInfo(undefined);
};


/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.rlenv.State.prototype.hasInfo = function() {
  return jspb.Message.getField(this, 1) != null;
};


/**
 * optional bytes legalActions = 2;
 * @return {!(string|Uint8Array)}
 */
proto.rlenv.State.prototype.getLegalactions = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 2, ""));
};


/**
 * optional bytes legalActions = 2;
 * This is a type-conversion wrapper around `getLegalactions()`
 * @return {string}
 */
proto.rlenv.State.prototype.getLegalactions_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getLegalactions()));
};


/**
 * optional bytes legalActions = 2;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getLegalactions()`
 * @return {!Uint8Array}
 */
proto.rlenv.State.prototype.getLegalactions_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getLegalactions()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.setLegalactions = function(value) {
  return jspb.Message.setProto3BytesField(this, 2, value);
};


/**
 * optional history.History history = 3;
 * @return {?proto.history.History}
 */
proto.rlenv.State.prototype.getHistory = function() {
  return /** @type{?proto.history.History} */ (
    jspb.Message.getWrapperField(this, history_pb.History, 3));
};


/**
 * @param {?proto.history.History|undefined} value
 * @return {!proto.rlenv.State} returns this
*/
proto.rlenv.State.prototype.setHistory = function(value) {
  return jspb.Message.setWrapperField(this, 3, value);
};


/**
 * Clears the message field making it undefined.
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.clearHistory = function() {
  return this.setHistory(undefined);
};


/**
 * Returns whether this field is set.
 * @return {boolean}
 */
proto.rlenv.State.prototype.hasHistory = function() {
  return jspb.Message.getField(this, 3) != null;
};


/**
 * optional bytes moveset = 4;
 * @return {!(string|Uint8Array)}
 */
proto.rlenv.State.prototype.getMoveset = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 4, ""));
};


/**
 * optional bytes moveset = 4;
 * This is a type-conversion wrapper around `getMoveset()`
 * @return {string}
 */
proto.rlenv.State.prototype.getMoveset_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getMoveset()));
};


/**
 * optional bytes moveset = 4;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getMoveset()`
 * @return {!Uint8Array}
 */
proto.rlenv.State.prototype.getMoveset_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getMoveset()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.setMoveset = function(value) {
  return jspb.Message.setProto3BytesField(this, 4, value);
};


/**
 * optional bytes team = 5;
 * @return {!(string|Uint8Array)}
 */
proto.rlenv.State.prototype.getTeam = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 5, ""));
};


/**
 * optional bytes team = 5;
 * This is a type-conversion wrapper around `getTeam()`
 * @return {string}
 */
proto.rlenv.State.prototype.getTeam_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getTeam()));
};


/**
 * optional bytes team = 5;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getTeam()`
 * @return {!Uint8Array}
 */
proto.rlenv.State.prototype.getTeam_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getTeam()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.setTeam = function(value) {
  return jspb.Message.setProto3BytesField(this, 5, value);
};


/**
 * optional string key = 6;
 * @return {string}
 */
proto.rlenv.State.prototype.getKey = function() {
  return /** @type {string} */ (jspb.Message.getFieldWithDefault(this, 6, ""));
};


/**
 * @param {string} value
 * @return {!proto.rlenv.State} returns this
 */
proto.rlenv.State.prototype.setKey = function(value) {
  return jspb.Message.setProto3StringField(this, 6, value);
};


goog.object.extend(exports, proto.rlenv);
