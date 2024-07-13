// source: history.proto
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

goog.exportSymbol('proto.history.ActionTypeEnum', null, global);
goog.exportSymbol('proto.history.History', null, global);
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
proto.history.History = function(opt_data) {
  jspb.Message.initialize(this, opt_data, 0, -1, null, null);
};
goog.inherits(proto.history.History, jspb.Message);
if (goog.DEBUG && !COMPILED) {
  /**
   * @public
   * @override
   */
  proto.history.History.displayName = 'proto.history.History';
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
proto.history.History.prototype.toObject = function(opt_includeInstance) {
  return proto.history.History.toObject(opt_includeInstance, this);
};


/**
 * Static version of the {@see toObject} method.
 * @param {boolean|undefined} includeInstance Deprecated. Whether to include
 *     the JSPB instance for transitional soy proto support:
 *     http://goto/soy-param-migration
 * @param {!proto.history.History} msg The msg instance to transform.
 * @return {!Object}
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.history.History.toObject = function(includeInstance, msg) {
  var f, obj = {
    active: msg.getActive_asB64(),
    boosts: msg.getBoosts_asB64(),
    sideconditions: msg.getSideconditions_asB64(),
    volatilestatus: msg.getVolatilestatus_asB64(),
    hyphenargs: msg.getHyphenargs_asB64(),
    weather: msg.getWeather_asB64(),
    pseudoweather: msg.getPseudoweather_asB64(),
    turncontext: msg.getTurncontext_asB64(),
    length: jspb.Message.getFieldWithDefault(msg, 9, 0)
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
 * @return {!proto.history.History}
 */
proto.history.History.deserializeBinary = function(bytes) {
  var reader = new jspb.BinaryReader(bytes);
  var msg = new proto.history.History;
  return proto.history.History.deserializeBinaryFromReader(msg, reader);
};


/**
 * Deserializes binary data (in protobuf wire format) from the
 * given reader into the given message object.
 * @param {!proto.history.History} msg The message object to deserialize into.
 * @param {!jspb.BinaryReader} reader The BinaryReader to use.
 * @return {!proto.history.History}
 */
proto.history.History.deserializeBinaryFromReader = function(msg, reader) {
  while (reader.nextField()) {
    if (reader.isEndGroup()) {
      break;
    }
    var field = reader.getFieldNumber();
    switch (field) {
    case 1:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setActive(value);
      break;
    case 2:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setBoosts(value);
      break;
    case 3:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setSideconditions(value);
      break;
    case 4:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setVolatilestatus(value);
      break;
    case 5:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setHyphenargs(value);
      break;
    case 6:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setWeather(value);
      break;
    case 7:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setPseudoweather(value);
      break;
    case 8:
      var value = /** @type {!Uint8Array} */ (reader.readBytes());
      msg.setTurncontext(value);
      break;
    case 9:
      var value = /** @type {number} */ (reader.readInt32());
      msg.setLength(value);
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
proto.history.History.prototype.serializeBinary = function() {
  var writer = new jspb.BinaryWriter();
  proto.history.History.serializeBinaryToWriter(this, writer);
  return writer.getResultBuffer();
};


/**
 * Serializes the given message to binary data (in protobuf wire
 * format), writing to the given BinaryWriter.
 * @param {!proto.history.History} message
 * @param {!jspb.BinaryWriter} writer
 * @suppress {unusedLocalVariables} f is only used for nested messages
 */
proto.history.History.serializeBinaryToWriter = function(message, writer) {
  var f = undefined;
  f = message.getActive_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      1,
      f
    );
  }
  f = message.getBoosts_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      2,
      f
    );
  }
  f = message.getSideconditions_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      3,
      f
    );
  }
  f = message.getVolatilestatus_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      4,
      f
    );
  }
  f = message.getHyphenargs_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      5,
      f
    );
  }
  f = message.getWeather_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      6,
      f
    );
  }
  f = message.getPseudoweather_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      7,
      f
    );
  }
  f = message.getTurncontext_asU8();
  if (f.length > 0) {
    writer.writeBytes(
      8,
      f
    );
  }
  f = message.getLength();
  if (f !== 0) {
    writer.writeInt32(
      9,
      f
    );
  }
};


/**
 * optional bytes active = 1;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getActive = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 1, ""));
};


/**
 * optional bytes active = 1;
 * This is a type-conversion wrapper around `getActive()`
 * @return {string}
 */
proto.history.History.prototype.getActive_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getActive()));
};


/**
 * optional bytes active = 1;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getActive()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getActive_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getActive()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setActive = function(value) {
  return jspb.Message.setProto3BytesField(this, 1, value);
};


/**
 * optional bytes boosts = 2;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getBoosts = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 2, ""));
};


/**
 * optional bytes boosts = 2;
 * This is a type-conversion wrapper around `getBoosts()`
 * @return {string}
 */
proto.history.History.prototype.getBoosts_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getBoosts()));
};


/**
 * optional bytes boosts = 2;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getBoosts()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getBoosts_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getBoosts()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setBoosts = function(value) {
  return jspb.Message.setProto3BytesField(this, 2, value);
};


/**
 * optional bytes sideConditions = 3;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getSideconditions = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 3, ""));
};


/**
 * optional bytes sideConditions = 3;
 * This is a type-conversion wrapper around `getSideconditions()`
 * @return {string}
 */
proto.history.History.prototype.getSideconditions_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getSideconditions()));
};


/**
 * optional bytes sideConditions = 3;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getSideconditions()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getSideconditions_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getSideconditions()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setSideconditions = function(value) {
  return jspb.Message.setProto3BytesField(this, 3, value);
};


/**
 * optional bytes volatileStatus = 4;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getVolatilestatus = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 4, ""));
};


/**
 * optional bytes volatileStatus = 4;
 * This is a type-conversion wrapper around `getVolatilestatus()`
 * @return {string}
 */
proto.history.History.prototype.getVolatilestatus_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getVolatilestatus()));
};


/**
 * optional bytes volatileStatus = 4;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getVolatilestatus()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getVolatilestatus_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getVolatilestatus()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setVolatilestatus = function(value) {
  return jspb.Message.setProto3BytesField(this, 4, value);
};


/**
 * optional bytes hyphenArgs = 5;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getHyphenargs = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 5, ""));
};


/**
 * optional bytes hyphenArgs = 5;
 * This is a type-conversion wrapper around `getHyphenargs()`
 * @return {string}
 */
proto.history.History.prototype.getHyphenargs_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getHyphenargs()));
};


/**
 * optional bytes hyphenArgs = 5;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getHyphenargs()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getHyphenargs_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getHyphenargs()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setHyphenargs = function(value) {
  return jspb.Message.setProto3BytesField(this, 5, value);
};


/**
 * optional bytes weather = 6;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getWeather = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 6, ""));
};


/**
 * optional bytes weather = 6;
 * This is a type-conversion wrapper around `getWeather()`
 * @return {string}
 */
proto.history.History.prototype.getWeather_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getWeather()));
};


/**
 * optional bytes weather = 6;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getWeather()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getWeather_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getWeather()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setWeather = function(value) {
  return jspb.Message.setProto3BytesField(this, 6, value);
};


/**
 * optional bytes pseudoweather = 7;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getPseudoweather = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 7, ""));
};


/**
 * optional bytes pseudoweather = 7;
 * This is a type-conversion wrapper around `getPseudoweather()`
 * @return {string}
 */
proto.history.History.prototype.getPseudoweather_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getPseudoweather()));
};


/**
 * optional bytes pseudoweather = 7;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getPseudoweather()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getPseudoweather_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getPseudoweather()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setPseudoweather = function(value) {
  return jspb.Message.setProto3BytesField(this, 7, value);
};


/**
 * optional bytes turnContext = 8;
 * @return {!(string|Uint8Array)}
 */
proto.history.History.prototype.getTurncontext = function() {
  return /** @type {!(string|Uint8Array)} */ (jspb.Message.getFieldWithDefault(this, 8, ""));
};


/**
 * optional bytes turnContext = 8;
 * This is a type-conversion wrapper around `getTurncontext()`
 * @return {string}
 */
proto.history.History.prototype.getTurncontext_asB64 = function() {
  return /** @type {string} */ (jspb.Message.bytesAsB64(
      this.getTurncontext()));
};


/**
 * optional bytes turnContext = 8;
 * Note that Uint8Array is not supported on all browsers.
 * @see http://caniuse.com/Uint8Array
 * This is a type-conversion wrapper around `getTurncontext()`
 * @return {!Uint8Array}
 */
proto.history.History.prototype.getTurncontext_asU8 = function() {
  return /** @type {!Uint8Array} */ (jspb.Message.bytesAsU8(
      this.getTurncontext()));
};


/**
 * @param {!(string|Uint8Array)} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setTurncontext = function(value) {
  return jspb.Message.setProto3BytesField(this, 8, value);
};


/**
 * optional int32 length = 9;
 * @return {number}
 */
proto.history.History.prototype.getLength = function() {
  return /** @type {number} */ (jspb.Message.getFieldWithDefault(this, 9, 0));
};


/**
 * @param {number} value
 * @return {!proto.history.History} returns this
 */
proto.history.History.prototype.setLength = function(value) {
  return jspb.Message.setProto3IntField(this, 9, value);
};


/**
 * @enum {number}
 */
proto.history.ActionTypeEnum = {
  MOVE: 0,
  SWITCH: 1
};

goog.object.extend(exports, proto.history);
